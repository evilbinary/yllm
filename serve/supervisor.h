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
    char leased_by[128];   /* 被哪个 server 租用(空 = 空闲) */
    uint64_t lease_expire; /* timed 策略到期时间戳(0 = permanent/未租) */
} SvNode;

typedef struct {
    SvNode nodes[SV_MAX_NODES];
    int n_nodes;
    /* 已拉起进程记录(node_id → pid, 自愈重拉前先杀旧进程)。
     * pid 为 spawn 的会话组长(sh -c 包装), kill(-pid) 连带 rank 子进程。 */
    char spawn_ids[SV_MAX_NODES][128];
    int spawn_pids[SV_MAX_NODES];
    int n_spawns;
    /* router 列表: 用统一 Node 抽象(type=router, addr="ip:port") */
    Node routers[SV_MAX_ROUTERS];
    int n_routers;
    char cache_dir[256];   /* 会话缓存落盘目录(传给 rank/server) */
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
    int64_t budget;         /* 传给 rank 的内存预算(MB; -1=auto) */
    volatile int quit;      /* 收到 QUIT 后退出主循环 */
    int no_spawn_server;    /* hub 合并模式: server 由本进程线程承担, 不 fork */
    ModelCfg models[CFG_MAX_MODELS];  /* 多模型: 每个模型一组 rank + 一个 server */
    int n_models;
} Supervisor;

/* yllm supervisor: 管理节点(收全部心跳 + 汇总 + 驱动 router) */
int cmd_supervisor(ServeConfig* cfg);
int supervisor_run(Supervisor* s);

#endif
