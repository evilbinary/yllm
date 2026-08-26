/* ctl.c — 管理命令客户端
 *   yllm ctl --config serve.yaml --target <rank-0|server|router|supervisor> --cmd <PING|STAT|DRAIN|QUIT|SCALE|QUERY_SERVERS> [--need-groups N]
 *
 * target → 端口映射(取自 serve.yaml):
 *   rank-<i>    -> rank_port_base + i
 *   server[-<i>] -> server_port + i
 *   router      -> router_port
 *   supervisor  -> sv_port
 */
#include "protocol.h"
#include "frame.h"
#include "sock.h"
#include "config.h"
#include "status.h"
#include "proclist.h"
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#else
#include <windows.h>
#include <process.h>
#define getpid() _getpid()
static void ysleep_ms(int ms) { Sleep((DWORD)ms); }
#endif
#ifndef _WIN32
static void ysleep_ms(int ms)
{
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
}
#endif
#include "../inference/include/log.h"
#include <stdio.h>
#include <string.h>

static int spawn_detached(const char* cmdline, const char* logfile)
{
#ifndef _WIN32
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setsid();
        if (logfile && logfile[0]) {
            int fd = open(logfile, O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); sock_close(fd); }
        }
        execl("/bin/sh", "sh", "-c", cmdline, (char*)NULL);
        _exit(127);
    }
    return (int)pid;
#else
    char* cmddup = _strdup(cmdline);
    if (!cmddup) return -1;
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    HANDLE logh = INVALID_HANDLE_VALUE;
    if (logfile && logfile[0]) {
        logh = CreateFileA(logfile, FILE_APPEND_DATA,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        si.hStdOutput = logh;
        si.hStdError = logh;
    } else {
        si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    }
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    int ok = CreateProcessA(NULL, cmddup, NULL, NULL, FALSE,
                            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                            NULL, NULL, &si, &pi);
    if (logh != INVALID_HANDLE_VALUE) CloseHandle(logh);
    if (!ok) { free(cmddup); return -1; }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    free(cmddup);
    return (int)pi.dwProcessId;
#endif
}

static int ctl_target_port(ServeConfig* cfg, const char* target)
{
    if (strcmp(target, "sv") == 0 || strcmp(target, "supervisor") == 0) return cfg->sv_port;
    if (strcmp(target, "rt") == 0 || strcmp(target, "router") == 0) return cfg->router_port;
    if ((target[0] == 'r' || target[0] == 's') && target[1] >= '0' && target[1] <= '9') {
        int idx = atoi(target + 1);
        if (target[0] == 'r') {
            /* rank-N: 全局连续编号, 端口 = base + N */
            return cfg->rank_port_base + idx;
        }
        /* server-N: 端口 = server_port + N*步长 */
        return cfg->server_port + idx * cfg_model_stride(cfg);
    }
    if (strncmp(target, "server", 6) == 0) {
        int idx = atoi(target + 6);
        return cfg->server_port + idx;
    }
    if (strncmp(target, "rank-", 5) == 0 || strncmp(target, "rank", 4) == 0) {
        int idx = atoi(target + 4);
        return cfg->rank_port_base + idx;
    }
    return -1;
}

/* 强制清理残留的 yllm 服务进程(hub/supervisor/router/server/rank),
 * 排除 ctl 自身与其他子命令(convert/chat/gen/dump 等)。
 * 跨平台: Linux /proc 扫描, Windows Toolhelp + PEB 命令行(proclist.h) */
static int kill_leftover_proc(int pid, const char* cmdline, void* ctx)
{
    const char* bin_hint = (const char*)ctx;
    if (pid <= 0 || pid == getpid()) return 0;
    if (!strstr(cmdline, "yllm")) return 0;
    if (strstr(cmdline, " ctl") || strstr(cmdline, " chat") || strstr(cmdline, " convert") ||
        strstr(cmdline, " gen ") || strstr(cmdline, " dump")) return 0;
    if (!strstr(cmdline, " hub ") && !strstr(cmdline, " supervisor ") &&
        !strstr(cmdline, " router ") && !strstr(cmdline, " server ") &&
        !strstr(cmdline, " rank ")) return 0;
    if (bin_hint && bin_hint[0] && !strstr(cmdline, bin_hint)) return 0;
    printf("exit: force-kill pid %d (%s)\n", pid, cmdline);
#ifdef _WIN32
    {
        HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
        if (h) { TerminateProcess(h, 1); CloseHandle(h); }
    }
#else
    kill(pid, SIGKILL);
#endif
    return 0;
}

static void kill_leftover_processes(const char* bin_hint)
{
    proclist_visit(kill_leftover_proc, (void*)bin_hint);
}

/* 向 host:port 发 DRAIN/QUIT, 打印一行结果。返回 0 成功联系上。 */
static int ctl_send_cmd(const char* label, const char* host, uint16_t port, const char* cmd)
{
    int fd = sock_connect(host, port, 1);
    if (fd < 0) {
        printf("%s: 不可达(已停止?) %s:%u\n", label, host, (unsigned)port);
        return -1;
    }
    frame_send(fd, cmd, NULL);
    Frame f;
    if (frame_recv(fd, &f) >= 0) printf("%s: %s %s\n", label, f.cmd, f.args);
    else printf("%s: 无响应\n", label);
    sock_close(fd);
    return 0;
}

/* 优先按 supervisor 节点表 DRAIN 所有 rank(含跨机); 失败则回退本机端口扫描 */
static void ctl_drain_ranks(ServeConfig* cfg)
{
    int drained = 0;
    int fd = sock_connect(cfg->sv_host, (uint16_t)cfg->sv_port, 1);
    if (fd >= 0) {
        frame_send(fd, PROTO_QUERY_SERVERS, NULL);
        Frame f;
        while (frame_recv(fd, &f) >= 0) {
            if (strcmp(f.cmd, PROTO_QUERY_DONE) == 0) break;
            if (strcmp(f.cmd, PROTO_SERVER_INFO) != 0) continue;
            char id[128] = "", type[16] = "", addr[128] = "";
            char vb[256];
            Frame ff;
            snprintf(ff.cmd, sizeof(ff.cmd), "X");
            snprintf(ff.args, sizeof(ff.args), "%s", f.args);
            sscanf(f.args, "%127s", id);
            if (frame_get(&ff, "type", vb, sizeof(vb)) == 0) snprintf(type, sizeof(type), "%s", vb);
            if (frame_get(&ff, "addr", vb, sizeof(vb)) == 0) snprintf(addr, sizeof(addr), "%s", vb);
            if (strcmp(type, "rank") != 0 || !addr[0]) continue;
            char* colon = strrchr(addr, ':');
            if (!colon) continue;
            *colon = '\0';
            uint16_t port = (uint16_t)atoi(colon + 1);
            if (ctl_send_cmd(id, addr, port, PROTO_DRAIN) == 0) drained++;
        }
        sock_close(fd);
    }
    if (drained > 0) return;

    /* supervisor 不可达或无 rank: 按 yaml 扫本机端口 */
    int nm = cfg->n_models > 0 ? cfg->n_models : 1;
    int stride = cfg_model_stride(cfg);
    int mi, r;
    for (mi = 0; mi < nm; mi++) {
        int ranks = mi < cfg->n_models && cfg->models[mi].ranks > 0
                    ? cfg->models[mi].ranks : (cfg->ranks > 0 ? cfg->ranks : 1);
        for (r = 0; r < ranks; r++) {
            int idx = mi * stride + r;
            char label[32];
            snprintf(label, sizeof(label), "rank-%d", idx);
            ctl_send_cmd(label, "127.0.0.1", (uint16_t)(cfg->rank_port_base + idx), PROTO_DRAIN);
        }
    }
}

/* stop: 只发优雅停机消息(DRAIN/QUIT), 绝不强杀。
 *   DRAIN 全部 rank(含远程, 先落盘) → DRAIN server → QUIT router → QUIT supervisor */
static int ctl_stop(ServeConfig* cfg)
{
    printf("stop: DRAIN ranks...\n");
    ctl_drain_ranks(cfg);

    printf("stop: DRAIN servers + QUIT router/supervisor...\n");
    {
        int nm = cfg->n_models > 0 ? cfg->n_models : 1;
        int stride = cfg_model_stride(cfg);
        int mi;
        for (mi = 0; mi < nm; mi++) {
            char label[32];
            snprintf(label, sizeof(label), "server-%d", mi);
            ctl_send_cmd(label, "127.0.0.1",
                         (uint16_t)(cfg->server_port + mi * stride), PROTO_DRAIN);
        }
    }
    ctl_send_cmd("router", "127.0.0.1", (uint16_t)cfg->router_port, PROTO_QUIT);
    ctl_send_cmd("supervisor", "127.0.0.1", (uint16_t)cfg->sv_port, PROTO_QUIT);
    return 0;
}

/* exit: 先 stop(只发消息落盘), 再兜底强杀残留 */
static int ctl_exit(ServeConfig* cfg)
{
    printf("exit: graceful stop (messages only)...\n");
    ctl_stop(cfg);

    printf("exit: wait for graceful shutdown...\n");
    ysleep_ms(3000);

    printf("exit: force-kill leftovers...\n");
    char bin_hint[128] = "";
    if (cfg->bin[0]) {
        const char* base = strrchr(cfg->bin, '/');
#ifdef _WIN32
        const char* base2 = strrchr(cfg->bin, '\\');
        if (!base || (base2 && base2 > base)) base = base2;
#endif
        snprintf(bin_hint, sizeof(bin_hint), "%s", base ? base + 1 : cfg->bin);
    }
    kill_leftover_processes(bin_hint);
    return 0;
}

/* 支持两种写法:
 *   yllm ctl --target rank-0 --cmd PING
 *   yllm ctl rank-0 PING      /   yllm ctl r0 PING   /   yllm ctl status
 */
int cmd_ctl(ServeConfig* cfg, int argc, char** argv)
{
    const char* target = NULL;
    const char* cmd = NULL;
    int need_groups = 0;
    const char* pos[3] = { NULL, NULL, NULL };
    int i, k;
    for (i = 2; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] == '-') {
            if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) target = argv[++i];
            else if (strcmp(argv[i], "--cmd") == 0 && i + 1 < argc) cmd = argv[++i];
            else if (strcmp(argv[i], "--need-groups") == 0 && i + 1 < argc) need_groups = atoi(argv[++i]);
            else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) i++; /* main 已加载, 跳过 */
        } else {
            for (k = 0; k < 3; k++) if (!pos[k]) { pos[k] = argv[i]; break; }
        }
    }
    /* `start <target>`: cmd 在前, target 在后(与 `target cmd` 相反) */
    if (pos[0] && strcmp(pos[0], "start") == 0) {
        cmd = "start";
        target = pos[1];
    } else if (!cmd) {
        if (pos[1]) {
            if (!target) target = pos[0];
            cmd = pos[1];
        } else {
            cmd = pos[0]; /* 单个位置参数: 视为 cmd, 如 `ctl status` */
        }
    }
    if (pos[2] && !(pos[0] && strcmp(pos[0], "start") == 0)) need_groups = atoi(pos[2]);
    if (!cmd) {
        fprintf(stderr, "usage: yllm ctl [--target <rank-0|r0|server|s0|router|rt|supervisor|sv>] "
                        "--cmd <PING|STAT|DRAIN|QUIT|SCALE|QUERY_SERVERS|status|start|stop|exit> [--need-groups N]\n");
        fprintf(stderr, "   or: yllm ctl <target> <cmd> [need]   e.g. yllm ctl r0 PING / yllm ctl status\n");
        return 1;
    }
    /* status: 完整状态查看(进程 + supervisor 节点表 + rank) */
    if (strcmp(cmd, "status") == 0) return cmd_status(cfg);

    /* start: 拉起角色进程(与 supervisor 相同的 spawn 方式, 常驻) */
    if (strcmp(cmd, "start") == 0) {
        if (!cfg->bin[0]) { fprintf(stderr, "ctl start: serve.yaml 缺 bin 字段\n"); return 1; }
        char cfgpath[512] = "";
        for (i = 2; i + 1 < argc; i++)
            if (strcmp(argv[i], "--config") == 0) snprintf(cfgpath, sizeof(cfgpath), "%s", argv[++i]);
        if (!cfgpath[0]) snprintf(cfgpath, sizeof(cfgpath), "%s", "serve.yaml");
        /* 日志目录: 取 cfg->log_file 的目录部分 */
        char logdir[512] = "logs";
        snprintf(logdir, sizeof(logdir), "%s", cfg->log_file[0] ? cfg->log_file : "logs/serve.log");
        {
            char* slash = strrchr(logdir, '/');
            if (slash) *slash = 0; else snprintf(logdir, sizeof(logdir), ".");
        }
        char cmd[2048];
        if (!target) {
            /* 默认: 拉起 hub(三合一, 自动带 rank) */
            snprintf(cmd, sizeof(cmd), "\"%s\" hub --config \"%s\"", cfg->bin, cfgpath);
        } else if (strcmp(target, "sv") == 0 || strcmp(target, "supervisor") == 0) {
            snprintf(cmd, sizeof(cmd), "\"%s\" supervisor --config \"%s\"", cfg->bin, cfgpath);
        } else if (strcmp(target, "rt") == 0 || strcmp(target, "router") == 0) {
            snprintf(cmd, sizeof(cmd), "\"%s\" router --config \"%s\"", cfg->bin, cfgpath);
        } else if (target[0] == 's' || strncmp(target, "server", 6) == 0) {
            int idx = target[0] == 's' ? atoi(target + 1) : atoi(target + 6);
            /* server-N: 模型 idx, 端口 = server_port + N*步长, leader = 该模型 rank0 */
            const char* mname = idx < cfg->n_models && cfg->models[idx].name[0]
                                ? cfg->models[idx].name : cfg->model_name;
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" server --id server-%d --model-name \"%s\" "
                     "--leader %s:%d --supervisor %s:%d --port %d --log %s/%s-server-%d.log",
                     cfg->bin, idx, mname[0] ? mname : "default",
                     cfg->sv_host, cfg->rank_port_base + idx * cfg_model_stride(cfg),
                     cfg->sv_host, cfg->sv_port,
                     cfg->server_port + idx * cfg_model_stride(cfg), logdir, mname, idx);
        } else if (target[0] == 'r' || strncmp(target, "rank", 4) == 0) {
            int idx = target[0] == 'r' ? atoi(target + 1) : atoi(target + 4);
            int st = cfg_model_stride(cfg);
            int mi = idx / st;
            int rr = idx % st;
            const char* model = mi < cfg->n_models && cfg->models[mi].model[0]
                                ? cfg->models[mi].model : cfg->model;
            const char* vocab = mi < cfg->n_models && cfg->models[mi].vocab[0]
                                ? cfg->models[mi].vocab : cfg->vocab;
            const char* mname = mi < cfg->n_models && cfg->models[mi].name[0]
                                ? cfg->models[mi].name : "default";
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" rank --model \"%s\" --vocab \"%s\" --port %d "
                     "--supervisor %s:%d --id rank-%d --log %s/%s-rank-%d.log",
                     cfg->bin, model, vocab, cfg->rank_port_base + idx,
                     cfg->sv_host, cfg->sv_port, idx, logdir, mname, rr);
        } else {
            fprintf(stderr, "ctl start: 未知目标 '%s'(支持 hub / sv / rt / s<N> / r<N>)\n", target);
            return 1;
        }
        char logfile[1024];
        snprintf(logfile, sizeof(logfile), "%s/ctl-start.log", logdir);
        int pid = spawn_detached(cmd, logfile);
        if (pid < 0) { fprintf(stderr, "ctl start: spawn 失败\n"); return 1; }
        printf("started %s (pid=%d, cmd: %s)\n", target ? target : "hub", pid, cmd);
        return 0;
    }

    /* stop: 只发 DRAIN/QUIT(落盘), 不强杀; exit: stop 后再强杀残留 */
    if (strcmp(cmd, "stop") == 0) return ctl_stop(cfg);
    if (strcmp(cmd, "exit") == 0) return ctl_exit(cfg);

    if (!target) target = "supervisor";
    int port = ctl_target_port(cfg, target);
    if (port < 0) { fprintf(stderr, "ctl: unknown target '%s'\n", target); return 1; }

    int fd = sock_connect("127.0.0.1", (uint16_t)port, 3);
    if (fd < 0) { fprintf(stderr, "ctl: %s not reachable on port %d\n", target, port); return 1; }

    char args[256];
    args[0] = 0;
    if (strcmp(cmd, "SCALE") == 0)
        snprintf(args, sizeof(args), "%s need_groups=%d", target, need_groups > 0 ? need_groups : 1);
    else if (strcmp(cmd, "QUERY_SERVERS") == 0)
        snprintf(args, sizeof(args), "%s", "");
    frame_send(fd, cmd, args[0] ? args : NULL);

    /* 读响应行, 直到连接关闭或收到 OK/ERR 后的一行 */
    Frame f;
    int n = 0;
    while (frame_recv(fd, &f) >= 0 && n < 64) {
        printf("%s %s\n", f.cmd, f.args);
        n++;
        if (strcmp(f.cmd, "OK") == 0 || strcmp(f.cmd, "ERR") == 0) break;
        if (strcmp(f.cmd, "QUERY_DONE") == 0) break;
    }
    sock_close(fd);
    return 0;
}
