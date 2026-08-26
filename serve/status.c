/* status.c — 查看 serve 集群状态
 *   yllm status --config serve.yaml
 *   输出: ① 本机 yllm 相关进程  ② supervisor 节点表(server)  ③ 各 rank 状态
 */
#include "protocol.h"
#include "frame.h"
#include "sock.h"
#include "node.h"
#include "config.h"
#include "proclist.h"
#include "../inference/include/log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int print_proc_one(int pid, const char* cmdline, void* ctx)
{
    int* shown = (int*)ctx;
    printf("  pid=%-7d %s\n", pid, cmdline);
    (*shown)++;
    return 0;
}

static void print_procs(void)
{
    printf("== 进程列表(本机 yllm 相关) ==\n");
    int shown = 0;
    proclist_visit(print_proc_one, &shown);
    if (!shown) printf("  (无 yllm 进程)\n");
}

#define NODE_MAX 32
typedef struct { int idx; char addr[128]; } NodeInfo;

static void dash_if_q(char* s)
{
    if (s[0] == '?' && s[1] == '\0') { s[0] = '-'; s[1] = 0; }
}

static void query_supervisor(ServeConfig* cfg, NodeInfo* ranks, int* n_rank, NodeInfo* srvs, int* n_srv)
{
    printf("== supervisor 节点表 (%s:%d) ==\n", cfg->sv_host, cfg->sv_port);
    printf("  %-10s %-7s %-12s %-7s %-22s %4s %6s\n",
           "id", "type", "model", "state", "addr", "in", "kv");
    int fd = sock_connect(cfg->sv_host, cfg->sv_port, 3);
    if (fd < 0) { printf("  (supervisor 不可达)\n"); return; }
    frame_send(fd, PROTO_QUERY_SERVERS, NULL);
    Frame f;
    int found = 0;
    while (frame_recv(fd, &f) >= 0) {
        if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) break;
        if (strcmp(f.cmd, PROTO_SERVER_INFO) == 0) {
            char id[128], type[16] = "-", model[128] = "-", state[64] = "-", addr[128] = "-";
            char s_inflight[64] = "-", s_kv[64] = "-";
            sscanf(f.args, "%127s", id);
            Frame ff;
            snprintf(ff.cmd, sizeof(ff.cmd), "X");
            snprintf(ff.args, sizeof(ff.args), "%s", f.args);
            char vb[256];
            if (frame_get(&ff, "type", vb, sizeof(vb)) == 0) snprintf(type, sizeof(type), "%s", vb);
            if (frame_get(&ff, "model", vb, sizeof(vb)) == 0) snprintf(model, sizeof(model), "%s", vb);
            if (frame_get(&ff, "state", vb, sizeof(vb)) == 0) snprintf(state, sizeof(state), "%s", vb);
            if (frame_get(&ff, "leader", vb, sizeof(vb)) == 0) snprintf(addr, sizeof(addr), "%s", vb);
            if (frame_get(&ff, "addr", vb, sizeof(vb)) == 0) snprintf(addr, sizeof(addr), "%s", vb);
            if (frame_get(&ff, "inflight", vb, sizeof(vb)) == 0) snprintf(s_inflight, sizeof(s_inflight), "%s", vb);
            if (frame_get(&ff, "kv_mb", vb, sizeof(vb)) == 0) snprintf(s_kv, sizeof(s_kv), "%s", vb);
            dash_if_q(model);
            dash_if_q(s_inflight);
            dash_if_q(s_kv);
            printf("  %-10s %-7s %-12s %-7s %-22s %4s %6s\n",
                   id, type, model, state, addr, s_inflight, s_kv);
            /* 收集 rank/server 编号+地址(数量不写死, 地址来自节点表) */
            if (strcmp(type, "rank") == 0 && *n_rank < NODE_MAX) {
                int idx = atoi(id + 5);
                if (strncmp(id, "rank-", 5) == 0) {
                    NodeInfo* ni = &ranks[(*n_rank)++];
                    ni->idx = idx;
                    snprintf(ni->addr, sizeof(ni->addr), "%s", addr);
                }
            } else if (strcmp(type, "server") == 0 && *n_srv < NODE_MAX) {
                int idx = atoi(id + 7);
                if (strncmp(id, "server-", 7) == 0) {
                    NodeInfo* ni = &srvs[(*n_srv)++];
                    ni->idx = idx;
                    snprintf(ni->addr, sizeof(ni->addr), "%s", addr);
                }
            }
            found = 1;
        }
    }
    sock_close(fd);
    if (!found) printf("  (无节点)\n");
}

static int stat_fetch(const char* addr, Frame* f)
{
    const char* colon = strchr(addr, ':');
    if (!colon) return -1;
    size_t hlen = (size_t)(colon - addr);
    int fd = sock_connect_host(addr, hlen, (uint16_t)atoi(colon + 1), 2);
    if (fd < 0) return -2;
    sock_set_timeout(fd, 3);
    frame_send(fd, PROTO_STAT, NULL);
    int rc = frame_recv(fd, f);
    sock_close(fd);
    return rc >= 0 ? 0 : -3;
}

static void stat_cpy(const char* args, const char* key, char* out, size_t n, const char* def)
{
    const char* p = strstr(args, key);
    if (!p) { snprintf(out, n, "%s", def); return; }
    p += strlen(key);
    if (*p == '=') p++;
    size_t i = 0;
    while (p[i] && p[i] != ' ' && i + 1 < n) {
        out[i] = p[i];
        i++;
    }
    out[i] = 0;
    if (out[0] == '?' || !out[0]) snprintf(out, n, "%s", def);
}

static int stat_int(const char* args, const char* key, int def)
{
    const char* p = strstr(args, key);
    if (!p) return def;
    p += strlen(key);
    if (*p == '=') p++;
    return atoi(p);
}

static unsigned long long stat_ull(const char* args, const char* key)
{
    const char* p = strstr(args, key);
    if (!p) return 0;
    p += strlen(key);
    if (*p == '=') p++;
    return strtoull(p, NULL, 10);
}

static void print_rank_header(void)
{
    printf("  %-8s %-4s %3s %3s %6s %-9s %3s %6s %6s %8s %6s %6s %6s\n",
           "id", "st", "in", "q", "kv", "layers", "omp", "pos", "need", "ms", "pf/s", "dc/s", "up");
}

static void query_one_rank(const char* label, const char* addr)
{
    const char* colon = strchr(addr, ':');
    if (!colon) {
        printf("  %-8s %-4s\n", label, "?");
        return;
    }
    size_t hlen = (size_t)(colon - addr);
    int fd = sock_connect_host(addr, hlen, (uint16_t)atoi(colon + 1), 2);
    if (fd < 0) {
        printf("  %-8s %-4s  (%s)\n", label, "--", addr);
        return;
    }
    sock_set_timeout(fd, 3);
    frame_send(fd, PROTO_STAT, NULL);
    Frame f;
    if (frame_recv(fd, &f) < 0) {
        printf("  %-8s %-4s  timeout (%s)\n", label, "--", addr);
        sock_close(fd);
        return;
    }
    sock_close(fd);
    const char* a = f.args;
    int inflight = stat_int(a, "inflight", 0);
    int queued = stat_int(a, "queued", 0);
    int omp_n = stat_int(a, "omp", 0);
    int jpos = stat_int(a, "job_pos", 0);
    int jneed = stat_int(a, "job_need", 0);
    unsigned long long jms = stat_ull(a, "job_ms");
    unsigned long long up = stat_ull(a, "uptime_s");
    float pft = 0, dct = 0;
    {
        const char* p = strstr(a, "job_pf_tps=");
        if (p) pft = (float)atof(p + 11);
        p = strstr(a, "job_dec_tps=");
        if (p) dct = (float)atof(p + 12);
    }
    char pfs[16], dcs[16];
    if (pft > 0.05f) snprintf(pfs, sizeof(pfs), "%.1f", pft);
    else snprintf(pfs, sizeof(pfs), "-");
    if (dct > 0.05f) snprintf(dcs, sizeof(dcs), "%.1f", dct);
    else snprintf(dcs, sizeof(dcs), "-");
    double kv = 0.0;
    {
        const char* p = strstr(a, "kv_mb=");
        if (p) kv = atof(p + 6);
    }
    char layers[16] = "-";
    {
        const char* p = strstr(a, "layers[");
        unsigned b = 0, e = 0;
        if (p && sscanf(p, "layers[%u,%u)", &b, &e) == 2)
            snprintf(layers, sizeof(layers), "[%u,%u)", b, e);
    }
    printf("  %-8s %-4s %3d %3d %6.1f %-9s %3d %6d %6d %8llu %6s %6s %6llu\n",
           label, f.cmd, inflight, queued, kv, layers, omp_n,
           jpos, jneed, (unsigned long long)jms, pfs, dcs, (unsigned long long)up);
}

static void query_ranks(ServeConfig* cfg, const NodeInfo* infos, int n)
{
    int r;
    print_rank_header();
    for (r = 0; r < n; r++) {
        char label[32];
        snprintf(label, sizeof(label), "rank-%d", infos[r].idx);
        char addr[128];
        if (infos[r].addr[0]) {
            snprintf(addr, sizeof(addr), "%s", infos[r].addr);
        } else {
            snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->rank_port_base + infos[r].idx);
        }
        query_one_rank(label, addr);
    }
}

static void query_servers(ServeConfig* cfg, const NodeInfo* infos, int n)
{
    int r;
    printf("  %-10s %-4s %3s %6s %6s %5s %-28s %-16s\n",
           "id", "st", "in", "kv", "up", "ranks", "peers", "ids");
    for (r = 0; r < n; r++) {
        char label[32], addr[128];
        snprintf(label, sizeof(label), "server-%d", infos[r].idx);
        if (infos[r].addr[0])
            snprintf(addr, sizeof(addr), "%s", infos[r].addr);
        else
            snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->server_port + infos[r].idx);
        Frame f;
        int rc = stat_fetch(addr, &f);
        if (rc != 0) {
            printf("  %-10s %-4s  (%s)\n", label, "--", addr);
            continue;
        }
        char peers[128], ids[64];
        stat_cpy(f.args, "lease_peers", peers, sizeof(peers), "-");
        stat_cpy(f.args, "lease_rank_ids", ids, sizeof(ids), "-");
        double kv = 0.0;
        const char* kp = strstr(f.args, "kv_mb=");
        if (kp) kv = atof(kp + 6);
        printf("  %-10s %-4s %3d %6.1f %6llu %5d %-28s %-16s\n",
               label, f.cmd,
               stat_int(f.args, "inflight", 0), kv,
               stat_ull(f.args, "uptime_s"),
               stat_int(f.args, "lease_ranks", 0),
               peers, ids);
    }
}

static void query_router(ServeConfig* cfg)
{
    char addr[128];
    snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->router_port);
    printf("  %-10s %-4s %7s %3s %6s %6s\n",
           "id", "st", "servers", "in", "kv", "up");
    Frame f;
    int rc = stat_fetch(addr, &f);
    if (rc != 0) {
        printf("  %-10s %-4s  (%s)\n", "router-0", "--", addr);
        return;
    }
    double kv = 0.0;
    const char* kp = strstr(f.args, "kv_mb=");
    if (kp) kv = atof(kp + 6);
    printf("  %-10s %-4s %7d %3d %6.1f %6llu\n",
           "router-0", f.cmd,
           stat_int(f.args, "servers", 0),
           stat_int(f.args, "inflight", 0), kv,
           stat_ull(f.args, "uptime_s"));
}

int cmd_status(ServeConfig* cfg)
{
    sock_init();
    NodeInfo ranks[NODE_MAX], srvs[NODE_MAX];
    int n_rank = 0, n_srv = 0;
    print_procs();
    printf("\n");
    query_supervisor(cfg, ranks, &n_rank, srvs, &n_srv);
    printf("\n");
    printf("== rank 状态 ==\n");
    if (n_rank > 0) {
        query_ranks(cfg, ranks, n_rank);
    } else {
        int cfg_ranks = cfg->ranks > 0 ? cfg->ranks : 1;
        int r;
        for (r = 0; r < cfg_ranks; r++) {
            NodeInfo ni; ni.idx = r; ni.addr[0] = 0;
            query_ranks(cfg, &ni, 1);
        }
    }
    printf("\n== server 状态 ==\n");
    if (n_srv > 0) {
        query_servers(cfg, srvs, n_srv);
    } else {
        int cfg_servers = cfg->servers > 0 ? cfg->servers : 1;
        int r;
        for (r = 0; r < cfg_servers; r++) {
            NodeInfo ni; ni.idx = r; ni.addr[0] = 0;
            query_servers(cfg, &ni, 1);
        }
    }
    printf("\n== router 状态 ==\n");
    query_router(cfg);
    return 0;
}
