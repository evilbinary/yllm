/* node.h — 统一节点结构(rank / server / router / supervisor 全部进程)
 *
 * 所有进程都是"一个 Node", 共享身份/心跳/状态语义, 只是 type 不同。
 * 心跳统一发往 supervisor(数据面); 判死/重拉归 supervisor(生命周期面)。
 */

#ifndef YLLM_SERVE_NODE_H
#define YLLM_SERVE_NODE_H

#include "frame.h"
#include <stdint.h>
#include <time.h>

#define NODE_STATE_LOADING 0
#define NODE_STATE_READY   1
#define NODE_STATE_BUSY    2   /* 推理中(rank: 处理 INFER; server: 转发中) */
#define NODE_STATE_DEAD    3

typedef struct {
    char node_id[128];
    char type[16];        /* rank | server | router | supervisor */
    char model[128];
    char addr[128];       /* 自身服务地址 ip:port */
    int state;            /* NODE_STATE_* */
    int inflight;
    double kv_mb;
    uint64_t last_hb;
    /* 心跳目标(supervisor) */
    char sv_host[128];
    uint16_t sv_port;
    int sv_enabled;
} Node;

/* 组装 HEARTBEAT 参数串 */
static inline int node_hb_args(const Node* n, char* out, size_t outsz)
{
    const char* st =
        n->state == NODE_STATE_READY ? "ready" :
        n->state == NODE_STATE_BUSY ? "busy" :
        n->state == NODE_STATE_DEAD ? "dead" : "loading";
    return snprintf(out, outsz,
                    "%s type=%s state=%s inflight=%d kv_mb=%.1f%s%s%s%s",
                    n->node_id, n->type, st,
                    n->inflight, n->kv_mb,
                    n->addr[0] ? " addr=" : "", n->addr[0] ? n->addr : "",
                    n->model[0] ? " model=" : "",
                    n->model[0] ? n->model : "");
}

/* 发送一次心跳到 supervisor(数据面: 活着就报) */
static inline void node_heartbeat(const Node* n)
{
    if (!n->sv_enabled) return;
    int fd = sock_connect(n->sv_host, n->sv_port, 3);
    if (fd < 0) return;
    char args[512];
    node_hb_args(n, args, sizeof(args));
    frame_send(fd, "HEARTBEAT", args);
    sock_close(fd);
}

/* 解析 HEARTBEAT args 进 Node(接收方: supervisor/router 用) */
static inline void node_parse_hb(const char* args, Node* out)
{
    Frame f;
    snprintf(f.cmd, sizeof(f.cmd), "HEARTBEAT");
    snprintf(f.args, sizeof(f.args), "%s", args);
    /* 首 token 是 node_id */
    const char* p = args;
    while (*p && *p != ' ') p++;
    size_t idlen = (size_t)(p - args);
    if (idlen >= sizeof(out->node_id)) idlen = sizeof(out->node_id) - 1;
    memcpy(out->node_id, args, idlen);
    out->node_id[idlen] = '\0';
    char v[256];
    if (frame_get(&f, "type", v, sizeof(v)) == 0) snprintf(out->type, sizeof(out->type), "%s", v);
    if (frame_get(&f, "model", v, sizeof(v)) == 0) snprintf(out->model, sizeof(out->model), "%s", v);
    if (frame_get(&f, "addr", v, sizeof(v)) == 0)  snprintf(out->addr, sizeof(out->addr), "%s", v);
    if (frame_get(&f, "state", v, sizeof(v)) == 0) {
        if (strcmp(v, "ready") == 0) out->state = NODE_STATE_READY;
        else if (strcmp(v, "busy") == 0) out->state = NODE_STATE_BUSY;
        else if (strcmp(v, "dead") == 0) out->state = NODE_STATE_DEAD;
        else out->state = NODE_STATE_LOADING;
    }
    if (frame_get(&f, "inflight", v, sizeof(v)) == 0) out->inflight = atoi(v);
    if (frame_get(&f, "kv_mb", v, sizeof(v)) == 0) out->kv_mb = atof(v);
    out->last_hb = (uint64_t)time(NULL);
}

/* 判死: 距上次心跳超过 timeout 秒 */
static inline int node_is_dead(const Node* n, int timeout)
{
    if (n->last_hb == 0) return 1;
    return (uint64_t)time(NULL) - n->last_hb > (uint64_t)timeout;
}

#endif /* YLLM_SERVE_NODE_H */
