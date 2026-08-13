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

static void query_supervisor(ServeConfig* cfg)
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
            found = 1;
        }
    }
    close(fd);
    if (!found) printf("  (无 server 节点)\n");
}

static void query_ranks(ServeConfig* cfg)
{
    printf("== rank 状态 (端口基址 %d) ==\n", cfg->rank_port_base);
    int r;
    for (r = 0; r < (cfg->ranks > 0 ? cfg->ranks : 1); r++) {
        int fd = sock_connect("127.0.0.1", (uint16_t)(cfg->rank_port_base + r), 2);
        if (fd < 0) { printf("  rank-%d: 不可达\n", r); continue; }
        frame_send(fd, PROTO_STAT, NULL);
        Frame f;
        if (frame_recv(fd, &f) >= 0)
            printf("  rank-%d: %s %s\n", r, f.cmd, f.args);
        close(fd);
    }
}

int cmd_status(ServeConfig* cfg)
{
    print_procs();
    printf("\n");
    query_supervisor(cfg);
    printf("\n");
    query_ranks(cfg);
    return 0;
}
