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
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "../inference/log.h"
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
            if (fd >= 0) { dup2(fd, 1); dup2(fd, 2); close(fd); }
        }
        execl("/bin/sh", "sh", "-c", cmdline, (char*)NULL);
        _exit(127);
    }
    return (int)pid;
#else
    char buf[4096];
    snprintf(buf, sizeof(buf), "start /b %s", cmdline);
    return system(buf);
#endif
}

static int ctl_target_port(ServeConfig* cfg, const char* target)
{
    if (strcmp(target, "sv") == 0 || strcmp(target, "supervisor") == 0) return cfg->sv_port;
    if (strcmp(target, "rt") == 0 || strcmp(target, "router") == 0) return cfg->router_port;
    if ((target[0] == 'r' || target[0] == 's') && target[1] >= '0' && target[1] <= '9') {
        int idx = atoi(target + 1);
        return target[0] == 'r' ? cfg->rank_port_base + idx : cfg->server_port + idx;
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
                        "--cmd <PING|STAT|DRAIN|QUIT|SCALE|QUERY_SERVERS|status|stop|start> [--need-groups N]\n");
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
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" server --id server-%d --model-name \"%s\" "
                     "--leader %s:%d --supervisor %s:%d --port %d --log %s/server-%d.log",
                     cfg->bin, idx, cfg->model_name[0] ? cfg->model_name : "default",
                     cfg->sv_host, cfg->rank_port_base, cfg->sv_host, cfg->sv_port,
                     cfg->server_port + idx, logdir, idx);
        } else if (target[0] == 'r' || strncmp(target, "rank", 4) == 0) {
            int idx = target[0] == 'r' ? atoi(target + 1) : atoi(target + 4);
            snprintf(cmd, sizeof(cmd),
                     "\"%s\" rank --model \"%s\" --vocab \"%s\" --port %d "
                     "--supervisor %s:%d --id rank-%d --log %s/rank-%d.log",
                     cfg->bin, cfg->model, cfg->vocab, cfg->rank_port_base + idx,
                     cfg->sv_host, cfg->sv_port, idx, logdir, idx);
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

    /* stop: 停止服务(所有 rank DRAIN, supervisor QUIT → hub 随之退出) */
    if (strcmp(cmd, "stop") == 0) {
        int r;
        for (r = 0; r < (cfg->ranks > 0 ? cfg->ranks : 1); r++) {
            int rfd = sock_connect("127.0.0.1", (uint16_t)(cfg->rank_port_base + r), 2);
            if (rfd >= 0) {
                frame_send(rfd, PROTO_DRAIN, NULL);
                Frame rf;
                if (frame_recv(rfd, &rf) >= 0) printf("rank-%d: %s %s\n", r, rf.cmd, rf.args);
                close(rfd);
            } else {
                printf("rank-%d: 不可达(已停止?)\n", r);
            }
        }
        int sfd = sock_connect("127.0.0.1", (uint16_t)cfg->sv_port, 2);
        if (sfd >= 0) {
            frame_send(sfd, PROTO_QUIT, NULL);
            Frame sf;
            if (frame_recv(sfd, &sf) >= 0) printf("supervisor: %s %s\n", sf.cmd, sf.args);
            close(sfd);
        } else {
            printf("supervisor: 不可达(已停止?)\n");
        }
        return 0;
    }

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
    close(fd);
    return 0;
}
