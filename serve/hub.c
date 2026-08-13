/* hub.c — yllm hub: 合并模式
 *
 * 把 supervisor + router + server 三个角色启动到同一个进程的不同线程,
 * 逻辑与端口完全不变(与分开模式一致), 只是省去三个独立进程。
 * rank 仍是独立进程(唯一加载模型权重的重进程)。
 *
 * 用法:
 *   yllm control --port 9500 --router-port 9400 --server-port 9420 \
 *                --server-id server-a --server-model tinyllama \
 *                --server-leader 127.0.0.1:9410
 */
#include "supervisor.h"
#include "router.h"
#include "server.h"
#include "../inference/yllm.h"
#include "../inference/log.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct { const char* key; const char* val; } ArgC;

static const char* opt_c(ArgC* args, int n, const char* key, const char* def)
{
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(args[i].key, key) == 0) return args[i].val;
    return def;
}

/* 线程包装: 跑 supervisor 主循环 */
typedef struct { Supervisor* s; } HubCtxSv;
static void sv_thread(void* arg)
{
    HubCtxSv* c = (HubCtxSv*)arg;
    ylog_info("hub: sv_thread enter");
    supervisor_run(c->s);
    ylog_info("hub: sv_thread exit");
}

typedef struct {
    Router* r;
    char sv_host[128];
    uint16_t sv_port;
} HubCtxRt;
static void rt_thread(void* arg)
{
    HubCtxRt* c = (HubCtxRt*)arg;
    ylog_info("hub: rt_thread enter");
    router_run(c->r, c->sv_host, c->sv_port);
    ylog_info("hub: rt_thread exit");
}

typedef struct { Server* s; } HubCtxSrv;
static void srv_thread(void* arg)
{
    HubCtxSrv* c = (HubCtxSrv*)arg;
    ylog_info("hub: srv_thread enter");
    server_run(c->s);
    ylog_info("hub: srv_thread exit");
}

int cmd_hub(int argc, char** argv)
{
    ArgC a[32];
    int n = 0;
    int i;
    for (i = 2; i + 1 < argc && n < 32; i += 2) {
        if (argv[i][0] != '-') break;
        a[n].key = argv[i];
        while (*a[n].key == '-') a[n].key++;
        a[n].val = argv[i + 1];
        n++;
    }
    const char* sv_port_s = opt_c(a, n, "port", "9500");
    const char* rt_port_s = opt_c(a, n, "router-port", "9400");
    const char* srv_port_s = opt_c(a, n, "server-port", "9420");
    const char* srv_id = opt_c(a, n, "server-id", "server-a");
    const char* srv_model = opt_c(a, n, "server-model", NULL);
    const char* srv_leader = opt_c(a, n, "server-leader", NULL);
    const char* strategy = opt_c(a, n, "strategy", "least");

    if (!srv_model || !srv_leader) {
        fprintf(stderr, "usage: yllm hub --port <sv> --router-port <rt> --server-port <srv> "
                        "--server-model <name> --server-leader <ip:port> [--server-id id] [--strategy s]\n");
        return 1;
    }

    /* supervisor */
    Supervisor sv;
    memset(&sv, 0, sizeof(sv));
    sv.port = (uint16_t)atoi(sv_port_s);
    /* router 地址(loopback, 走网络通知, 逻辑不变) */
    Node* rn = &sv.routers[sv.n_routers++];
    memset(rn, 0, sizeof(*rn));
    snprintf(rn->node_id, sizeof(rn->node_id), "router-0");
    snprintf(rn->type, sizeof(rn->type), "router");
    snprintf(rn->addr, sizeof(rn->addr), "127.0.0.1:%s", rt_port_s);

    /* router */
    Router rt;
    memset(&rt, 0, sizeof(rt));
    rt.port = (uint16_t)atoi(rt_port_s);
    rt.strategy = strategy;
    snprintf(rt.node.node_id, sizeof(rt.node.node_id), "router-0");
    snprintf(rt.node.type, sizeof(rt.node.type), "router");
    rt.node.state = NODE_STATE_READY;
    char sv_host[128] = "127.0.0.1";
    uint16_t sv_port = (uint16_t)atoi(sv_port_s);

    /* server */
    Server srv;
    memset(&srv, 0, sizeof(srv));
    snprintf(srv.node.node_id, sizeof(srv.node.node_id), "%s", srv_id);
    snprintf(srv.node.type, sizeof(srv.node.type), "server");
    snprintf(srv.node.model, sizeof(srv.node.model), "%s", srv_model);
    srv.node.state = NODE_STATE_READY;
    srv.port = (uint16_t)atoi(srv_port_s);
    snprintf(srv.node.addr, sizeof(srv.node.addr), "127.0.0.1:%s", srv_port_s);
    /* leader rank 地址 */
    const char* colon = strchr(srv_leader, ':');
    if (!colon) { fprintf(stderr, "bad leader addr: %s\n", srv_leader); return 1; }
    size_t hlen = (size_t)(colon - srv_leader);
    if (hlen >= sizeof(srv.leader_host)) hlen = sizeof(srv.leader_host) - 1;
    memcpy(srv.leader_host, srv_leader, hlen);
    srv.leader_host[hlen] = '\0';
    srv.leader_port = (uint16_t)atoi(colon + 1);
    /* server 心跳发本进程 supervisor(loopback) */
    snprintf(srv.node.sv_host, sizeof(srv.node.sv_host), "127.0.0.1");
    srv.node.sv_port = sv_port;
    srv.node.sv_enabled = 1;

    ylog_info("control: merged mode (sv=%u rt=%u srv=%u leader=%s:%u)",
              sv.port, rt.port, srv.port, srv.leader_host, srv.leader_port);

    void* t1 = NULL, *t2 = NULL, *t3 = NULL;
    HubCtxSv c1; c1.s = &sv;
    HubCtxRt c2; c2.r = &rt; snprintf(c2.sv_host, sizeof(c2.sv_host), "%s", sv_host); c2.sv_port = sv_port;
    HubCtxSrv c3; c3.s = &srv;
    ythread_create(&t1, sv_thread, &c1);
    ythread_create(&t2, rt_thread, &c2);
    ythread_create(&t3, srv_thread, &c3);
    ythread_join(&t1);
    ythread_join(&t2);
    ythread_join(&t3);
    return 0;
}
