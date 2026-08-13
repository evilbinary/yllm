#ifndef YLLM_SERVE_SUPERVISOR_H
#define YLLM_SERVE_SUPERVISOR_H

#include "node.h"

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
} Supervisor;

/* yllm supervisor: 管理节点(收全部心跳 + 汇总 + 驱动 router) */
int cmd_supervisor(int argc, char** argv);
int supervisor_run(Supervisor* s);

#endif
