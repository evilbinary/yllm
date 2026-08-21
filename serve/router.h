#ifndef YLLM_SERVE_ROUTER_H
#define YLLM_SERVE_ROUTER_H

#include "node.h"
#include "config.h"
#include <pthread.h>

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
    pthread_mutex_t lock;    /* 保护 servers[] 表(supervisor 通知线程 vs 请求线程) */
    volatile int quit;     /* 收到 QUIT 后退出主循环 */
    uint16_t port;
    int rr_counter;
    const char* strategy;
    Node node;           /* 自身节点(type=router), 心跳发 supervisor */
} Router;

/* yllm router: 调度层(server 注册表 + 路由决策 + 请求转发) */
int cmd_router(ServeConfig* cfg);
int router_run(Router* r, const char* sv_host, uint16_t sv_port);

/* 通用 INFER 转发(HTTP 层等复用): 路由到 server → 转发 → 逐 token 回调。
 * 返回: 0 成功, -2 模型不存在, -1 失败; err/errlen 可空, 失败时写入后端 ERR 正文。 */
int router_infer(Router* r, const char* model, int max_tokens,
                 const char* prompt, size_t plen,
                 void (*on_token)(const char* utf8, size_t len, void* ctx), void* ctx,
                 float temp, float top_p, char* err, size_t errlen);

/* 会话模式推理(转发): 带会话 key + 新消息文本发给 server(会话管理在 server 侧),
 * server 渲染/缓存后只把增量 token 发给 rank。on_token 同 router_infer。
 * prompt_tokens(可空): 传出本请求 prompt 总 token 数(server 真实渲染统计)。
 * err/errlen(可空): 失败时写入后端 ERR 正文(已剥 v=)。 */
int router_infer_sess(Router* r, const char* model, int max_tokens,
                      const char* sess_key, const char* new_msg, size_t msg_len,
                      void (*on_token)(const char* utf8, size_t len, void* ctx), void* ctx,
                      float temp, float top_p, int* prompt_tokens,
                      char* err, size_t errlen);

#endif
