/* status.c — 查看 serve 集群状态
 *   yllm status --config serve.yaml
 *   输出: ① 本机 yllm 相关进程  ② supervisor 节点表(server)  ③ 各 rank 状态
 */
#include "protocol.h"
#include "frame.h"
#include "sock.h"
#include "node.h"
#include "config.h"
#include "../inference/log.h"
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#else
#include <dirent.h>
#endif

static void print_procs(void)
{
    printf("== 进程列表(本机 yllm 相关) ==\n");
#ifdef _WIN32
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE) return;
    PROCESSENTRY32W pe;
    pe.dwSize = sizeof(pe);
    if (Process32FirstW(snap, &pe)) {
        do {
            if (wcsstr(pe.szExeFile, L"yllm") != NULL)
                printf("  pid=%-7lu %ls\n", (unsigned long)pe.th32ProcessID, pe.szExeFile);
        } while (Process32NextW(snap, &pe));
    }
    CloseHandle(snap);
#else
    DIR* d = opendir("/proc");
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[64], cmd[4096];
        snprintf(path, sizeof(path), "/proc/%s/cmdline", e->d_name);
        FILE* f = fopen(path, "rb");
        if (!f) continue;
        size_t n = fread(cmd, 1, sizeof(cmd) - 1, f);
        fclose(f);
        if (n == 0) continue;
        cmd[n] = 0;
        /* cmdline 以 \0 分隔, 转空格便于显示 */
        size_t i;
        for (i = 0; i < n; i++) if (cmd[i] == 0) cmd[i] = ' ';
        if (strstr(cmd, "yllm") == NULL) continue;
        printf("  pid=%-7s %s\n", e->d_name, cmd);
    }
    closedir(d);
#endif
}

#define NODE_MAX 32
typedef struct { int idx; char addr[128]; } NodeInfo;
static void query_supervisor(ServeConfig* cfg, NodeInfo* ranks, int* n_rank, NodeInfo* srvs, int* n_srv)
{
    printf("== supervisor 节点表 (%s:%d) ==\n", cfg->sv_host, cfg->sv_port);
    int fd = sock_connect(cfg->sv_host, cfg->sv_port, 3);
    if (fd < 0) { printf("  (supervisor 不可达)\n"); return; }
    frame_send(fd, PROTO_QUERY_SERVERS, NULL);
    Frame f;
    int found = 0;
    while (frame_recv(fd, &f) >= 0) {
        if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) break;
        if (strcmp(f.cmd, PROTO_SERVER_INFO) == 0) {
            char id[128], type[16] = "?", model[128] = "?", state[64] = "?", addr[128] = "?";
            char s_inflight[64] = "?", s_kv[64] = "?";
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
            printf("  %-10s %-6s model=%-12s state=%-7s addr=%-22s inflight=%s kv_mb=%s\n",
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
    close(fd);
    if (!found) printf("  (无 server 节点)\n");
}

static void query_role(const char* label, const char* addr)
{
    const char* colon = strchr(addr, ':');
    if (!colon) { printf("  %s: 地址无效 %s\n", label, addr); return; }
    size_t hlen = (size_t)(colon - addr);
    int fd = sock_connect_host(addr, hlen, (uint16_t)atoi(colon + 1), 2);
    if (fd < 0) { printf("  %s: 不可达 (%s)\n", label, addr); return; }
    frame_send(fd, PROTO_STAT, NULL);
    Frame f;
    if (frame_recv(fd, &f) >= 0)
        printf("  %s: %s %s\n", label, f.cmd, f.args);
    close(fd);
}

static void query_ranks(ServeConfig* cfg, const NodeInfo* infos, int n)
{
    int r;
    for (r = 0; r < n; r++) {
        char label[32];
        snprintf(label, sizeof(label), "rank-%d", infos[r].idx);
        char addr[128];
        if (infos[r].addr[0]) {
            snprintf(addr, sizeof(addr), "%s", infos[r].addr);
        } else {
            snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->rank_port_base + infos[r].idx);
        }
        query_role(label, addr);
    }
}

static void query_servers(ServeConfig* cfg, const NodeInfo* infos, int n)
{
    int r;
    for (r = 0; r < n; r++) {
        char label[32];
        snprintf(label, sizeof(label), "server-%d", infos[r].idx);
        char addr[128];
        if (infos[r].addr[0]) {
            snprintf(addr, sizeof(addr), "%s", infos[r].addr);
        } else {
            snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->server_port + infos[r].idx);
        }
        query_role(label, addr);
    }
}

int cmd_status(ServeConfig* cfg)
{
    NodeInfo ranks[NODE_MAX], srvs[NODE_MAX];
    int n_rank = 0, n_srv = 0;
    print_procs();
    printf("\n");
    query_supervisor(cfg, ranks, &n_rank, srvs, &n_srv);
    printf("\n");
    printf("== rank 状态 (端口基址 %d) ==\n", cfg->rank_port_base);
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
    {
        char addr[128];
        snprintf(addr, sizeof(addr), "127.0.0.1:%d", cfg->router_port);
        query_role("router-0", addr);
    }
    return 0;
}
