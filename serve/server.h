#ifndef YLLM_SERVE_SERVER_H
#define YLLM_SERVE_SERVER_H

#include "node.h"
#include "config.h"

typedef struct {
    Node node;             /* 统一节点身份(type=server) */
    char leader_host[128]; /* 租用的 leader rank(rank0) 地址 */
    uint16_t leader_port;
    uint16_t port;         /* server 自身监听端口(router 转发入口) */
    uint64_t start_s;
    volatile int quit;     /* 收到 QUIT/DRAIN 后退出主循环 */
    char resolve_model[128]; /* 租用/发现查询用的模型(llf 路径, 与 rank 心跳上报一致) */
    char lease_strategy[32]; /* request(每次推理租/释) | timed | permanent */
    int  lease_duration;     /* timed 租期(秒) */
    int  lease_ranks;        /* 租到的组内 rank 数 */
    char lease_peers[1024];  /* 组内各段节点 IP(逗号分隔, 段号顺序, 随 INFER 捎给 rank0) */
    char lease_rank_ids[512]; /* 租用组内 rank 的 id 列表(rank-0,rank-1), 状态展示用 */
} Server;

/* yllm server: 业务逻辑组(租用 rank 组, 转发请求, 广播注册/心跳) */
int cmd_server(ServeConfig* cfg);
int server_run(Server* s);

/* 自动发现 leader: 向 supervisor 查询该模型的 ready rank(填 leader_host/port)。
 * 返回 0 成功, -1 失败(无 rank / 查询失败)。 */
int server_resolve_leader(Server* s, const char* sv_host, uint16_t sv_port, const char* model);

#endif
