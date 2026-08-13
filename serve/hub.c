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

int cmd_hub(ServeConfig* cfg)
{
    if (!cfg->model_name[0] || !cfg->leader[0]) {
        fprintf(stderr, "usage: yllm hub --server-model <name> --server-leader <ip:port> "
                        "[--port sv] [--router-port rt] [--server-port srv] [--http-port N]\n");
        return 1;
    }

    /* supervisor */
    Supervisor sv;
    memset(&sv, 0, sizeof(sv));
    sv.port = (uint16_t)cfg->sv_port;
    sv.ranks = cfg->ranks > 0 ? cfg->ranks : 1;
    sv.rank_port_base = (uint16_t)cfg->rank_port_base;
    sv.server_port_base = (uint16_t)cfg->server_port;
    sv.auto_heal = cfg->auto_heal;
    sv.no_spawn_server = 1; /* hub 内已有 server 线程 */
    snprintf(sv.sv_host, sizeof(sv.sv_host), "%s", cfg->sv_host);
    snprintf(sv.bin, sizeof(sv.bin), "%s", cfg->bin);
    if (cfg->model[0]) snprintf(sv.model, sizeof(sv.model), "%s", cfg->model);
    if (cfg->vocab[0]) snprintf(sv.vocab, sizeof(sv.vocab), "%s", cfg->vocab);
    if (cfg->model_name[0]) snprintf(sv.model_name, sizeof(sv.model_name), "%s", cfg->model_name);
    /* router 地址(loopback, 走网络通知, 逻辑不变) */
    Node* rn = &sv.routers[sv.n_routers++];
    memset(rn, 0, sizeof(*rn));
    snprintf(rn->node_id, sizeof(rn->node_id), "router-0");
    snprintf(rn->type, sizeof(rn->type), "router");
    snprintf(rn->addr, sizeof(rn->addr), "127.0.0.1:%d", cfg->router_port);

    /* router */
    Router rt;
    memset(&rt, 0, sizeof(rt));
    rt.port = (uint16_t)cfg->router_port;
    rt.strategy = cfg->strategy;
    snprintf(rt.node.node_id, sizeof(rt.node.node_id), "router-0");
    snprintf(rt.node.type, sizeof(rt.node.type), "router");
    rt.node.state = NODE_STATE_READY;
    char sv_host[128] = "127.0.0.1";
    uint16_t sv_port = (uint16_t)cfg->sv_port;

    /* server */
    Server srv;
    memset(&srv, 0, sizeof(srv));
    snprintf(srv.node.node_id, sizeof(srv.node.node_id), "server-0");
    snprintf(srv.node.type, sizeof(srv.node.type), "server");
    snprintf(srv.node.model, sizeof(srv.node.model), "%s", cfg->model_name);
    srv.node.state = NODE_STATE_READY;
    srv.port = (uint16_t)cfg->server_port;
    snprintf(srv.node.addr, sizeof(srv.node.addr), "127.0.0.1:%d", cfg->server_port);
    /* leader rank 地址 */
    const char* colon = strchr(cfg->leader, ':');
    if (!colon) { fprintf(stderr, "bad leader addr: %s\n", cfg->leader); return 1; }
    size_t hlen = (size_t)(colon - cfg->leader);
    if (hlen >= sizeof(srv.leader_host)) hlen = sizeof(srv.leader_host) - 1;
    memcpy(srv.leader_host, cfg->leader, hlen);
    srv.leader_host[hlen] = '\0';
    srv.leader_port = (uint16_t)atoi(colon + 1);
    /* server 心跳发本进程 supervisor(loopback) */
    snprintf(srv.node.sv_host, sizeof(srv.node.sv_host), "127.0.0.1");
    srv.node.sv_port = sv_port;
    srv.node.sv_enabled = 1;

    ylog_info("hub: merged mode (sv=%u rt=%u srv=%u leader=%s:%u)",
              sv.port, rt.port, srv.port, srv.leader_host, srv.leader_port);

    void* t1 = NULL, *t2 = NULL, *t3 = NULL;
    HubCtxSv c1; c1.s = &sv;
    HubCtxRt c2; c2.r = &rt; snprintf(c2.sv_host, sizeof(c2.sv_host), "%s", sv_host); c2.sv_port = sv_port;
    HubCtxSrv c3; c3.s = &srv;
    if (cfg->http_port > 0) {
        extern int router_http_start(Router* r, uint16_t http_port);
        router_http_start(&rt, (uint16_t)cfg->http_port);
    }
    ythread_create(&t1, sv_thread, &c1);
    ythread_create(&t2, rt_thread, &c2);
    ythread_create(&t3, srv_thread, &c3);
    ythread_join(&t1);
    ythread_join(&t2);
    ythread_join(&t3);
    return 0;
}
