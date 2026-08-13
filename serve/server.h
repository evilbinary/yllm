#ifndef YLLM_SERVE_SERVER_H
#define YLLM_SERVE_SERVER_H

#include "node.h"
#include "config.h"

typedef struct {
    Node node;             /* 统一节点身份(type=server) */
    char leader_host[128];
    uint16_t leader_port;
    uint16_t port;         /* server 自身监听端口(router 转发入口) */
    uint64_t start_s;
    volatile int quit;     /* 收到 QUIT/DRAIN 后退出主循环 */
} Server;

/* yllm server: 业务逻辑组(租用 rank 组, 转发请求, 广播注册/心跳) */
int cmd_server(ServeConfig* cfg);
int server_run(Server* s);

#endif
