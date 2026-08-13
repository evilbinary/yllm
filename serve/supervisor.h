#ifndef YLLM_SERVE_SUPERVISOR_H
#define YLLM_SERVE_SUPERVISOR_H

#include "node.h"
#include "config.h"

#define SV_MAX_NODES 256
#define SV_MAX_ROUTERS 16
#define SV_HB_TIMEOUT 10

typedef struct {
    Node node;
    int router_notified;   /* 已通知 router(server ADD/DEL 去重) */
} SvNode;

typedef struct {
    SvNode nodes[SV_MAX_NODES];
    int n_nodes;
    /* router 列表: 用统一 Node 抽象(type=router, addr="ip:port") */
    Node routers[SV_MAX_ROUTERS];
    int n_routers;
    uint16_t port;
    /* 生命周期配置(拉起/自愈用) */
    char bin[512];          /* yllm 二进制路径 */
    char model[512];        /* 模型 llf 路径 */
    char vocab[512];        /* vocab 路径 */
    char model_name[128];   /* server 注册的模型名 */
    int ranks;              /* 每 server 的 rank 段数 */
    uint16_t rank_port_base;  /* rank 端口基址 */
    uint16_t server_port_base;/* server 端口基址 */
    char sv_host[128];      /* supervisor 自身地址(rank/server 心跳目标) */
    int auto_heal;          /* 自愈开关 */
} Supervisor;

/* yllm supervisor: 管理节点(收全部心跳 + 汇总 + 驱动 router) */
int cmd_supervisor(ServeConfig* cfg);
int supervisor_run(Supervisor* s);

#endif
