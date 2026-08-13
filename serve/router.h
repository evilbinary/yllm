#ifndef YLLM_SERVE_ROUTER_H
#define YLLM_SERVE_ROUTER_H

#include "node.h"

#define RT_MAX_SERVERS 64

typedef struct {
    char id[128];
    char model[128];
    char leader_host[128];
    uint16_t leader_port;
    int state;           /* NODE_STATE_* */
    int inflight;
    double kv_mb;
    uint64_t last_update;
} RtServer;

typedef struct {
    RtServer servers[RT_MAX_SERVERS];
    int n_servers;
    uint16_t port;
    int rr_counter;
    const char* strategy;
    Node node;           /* 自身节点(type=router), 心跳发 supervisor */
} Router;

/* yllm router: 调度层(server 注册表 + 路由决策 + 请求转发) */
int cmd_router(int argc, char** argv);
int router_run(Router* r, const char* sv_host, uint16_t sv_port);

/* 通用 INFER 转发(HTTP 层等复用): 路由到 server → 转发 → 逐 token 回调 */
int router_infer(Router* r, const char* model, int max_tokens,
                 const char* prompt, size_t plen,
                 void (*on_token)(const char* utf8, size_t len, void* ctx), void* ctx);

#endif
