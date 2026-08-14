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
    if (cfg->n_models == 0 && !cfg->model[0]) {
        fprintf(stderr, "usage: yllm hub --config serve.yaml (serve.yaml 需含模型配置)\n");
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
    sv.n_models = cfg->n_models;
    {
        int mi;
        for (mi = 0; mi < cfg->n_models && mi < CFG_MAX_MODELS; mi++) sv.models[mi] = cfg->models[mi];
    }
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
    pthread_mutex_init(&rt.lock, NULL);
    rt.strategy = cfg->strategy;
    snprintf(rt.node.node_id, sizeof(rt.node.node_id), "router-0");
    snprintf(rt.node.type, sizeof(rt.node.type), "router");
    rt.node.state = NODE_STATE_READY;
    char sv_host[128] = "127.0.0.1";
    uint16_t sv_port = (uint16_t)cfg->sv_port;

    /* 每个模型一个 server 线程(端口 = server_port + mi*16, leader = 该模型 rank0) */
    Server srv[CFG_MAX_MODELS];
    HubCtxSrv c3[CFG_MAX_MODELS];
    void* t3[CFG_MAX_MODELS];
    int mi;
    if (cfg->n_models == 0) cfg->n_models = 1;
    for (mi = 0; mi < cfg->n_models; mi++) {
        ModelCfg* mc = &cfg->models[mi];
        Server* s = &srv[mi];
        memset(s, 0, sizeof(*s));
        s->start_s = (uint64_t)time(NULL);
        snprintf(s->node.node_id, sizeof(s->node.node_id), "server-%d", mi);
        snprintf(s->node.type, sizeof(s->node.type), "server");
        snprintf(s->node.model, sizeof(s->node.model), "%s",
                 mc->name[0] ? mc->name : (cfg->model_name[0] ? cfg->model_name : "default"));
        s->node.state = NODE_STATE_READY;
        s->port = (uint16_t)(cfg->server_port + mi * cfg_model_stride(cfg));
        snprintf(s->node.addr, sizeof(s->node.addr), "127.0.0.1:%d", s->port);
        /* leader = 该模型 rank0 (rank_port_base + mi*16); 主机由 server_run 自动发现(查询 supervisor 注册表) */
        if (mc->model[0])
            snprintf(s->resolve_model, sizeof(s->resolve_model), "%s", mc->model);
        s->leader_port = (uint16_t)(cfg->rank_port_base + mi * cfg_model_stride(cfg));
        snprintf(s->node.sv_host, sizeof(s->node.sv_host), "127.0.0.1");
        s->node.sv_port = sv_port;
        s->node.sv_enabled = 1;
        snprintf(s->lease_strategy, sizeof(s->lease_strategy), "%s",
                 cfg->lease_strategy[0] ? cfg->lease_strategy : "request");
        s->lease_duration = cfg->lease_duration;
        c3[mi].s = s;
        ylog_info("hub: server-%d model=%s port=%d leader=%s:%d",
                  mi, s->node.model, s->port, s->leader_host, s->leader_port);
    }

    ylog_info("hub: merged mode (sv=%u rt=%u srv=%u model(s)=%d)",
              sv.port, rt.port, cfg->server_port, cfg->n_models);

    void* t1 = NULL, *t2 = NULL;
    HubCtxSv c1; c1.s = &sv;
    HubCtxRt c2; c2.r = &rt; snprintf(c2.sv_host, sizeof(c2.sv_host), "%s", sv_host); c2.sv_port = sv_port;
    if (cfg->http_port > 0) {
        extern int router_http_start(Router* r, uint16_t http_port);
        router_http_start(&rt, (uint16_t)cfg->http_port);
    }
    ythread_create(&t1, sv_thread, &c1);
    ythread_create(&t2, rt_thread, &c2);
    for (mi = 0; mi < cfg->n_models; mi++)
        ythread_create(&t3[mi], srv_thread, &c3[mi]);
    ythread_join(&t1);
    ythread_join(&t2);
    for (mi = 0; mi < cfg->n_models; mi++)
        ythread_join(&t3[mi]);
    return 0;
}
