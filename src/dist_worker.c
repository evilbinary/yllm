/* dist_worker.c — 常驻分布式 worker
 *
 * 局域网多节点场景: 管理节点通过 TCP 控制端口向各节点下发推理任务,
 * 收到命令后本节点后台拉起 `yllm gen --rank ...` 子进程, 输出写入日志。
 * 这样 make dist 可在无互相 SSH 的条件下编排各 rank。
 *
 * 服务端(在业务节点常驻):
 *   dist-worker --port <ctl_port> [--bin <yllm路径>] [--logdir <目录>]
 *
 * 文件中转源(在管理节点跑一个即可, 私有 TCP 帧):
 *   dist-worker --serve <port> --root <目录>
 *
 * 客户端(管理节点发单条命令):
 *   dist-worker --host <ip> --port <ctl_port> --send "<命令行>"
 *
 * 轻量文本协议, 一条命令一行, 回复一行("ok" 或 "err <msg>"):
 *   ping                        探测存活
 *   run RANK RANKS PB TOKENS MODEL VOCAB PROMPT... 拉起 gen 子进程 (prompt 为行尾剩余部分)
 *   sync HOST PORT REL DEST     从 HOST:PORT 拉 REL(相对 serve root) 到本地 DEST,
 *                               已有且 size+mtime 一致则跳过
 *   stop                        终止本 worker 曾启动的全部子进程
 *   quit                        退出 worker
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>
#define DBG(fmt, ...) do { fprintf(stderr, fmt "\n", ##__VA_ARGS__); fflush(stderr); } while (0)
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <sys/stat.h>
#include <utime.h>
#define close(fd) closesocket(fd)
#define ssize_t int
#define stat_ok(st) ((st).st_size)
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <signal.h>
#include <utime.h>
#endif

#define MAX_LINE 8192
#define MAX_CHILDREN 64

static int g_children[MAX_CHILDREN];
static int g_nchildren = 0;

#ifdef _WIN32
static void ws_init(void)
{
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
}
#endif

static void child_add(int pid)
{
    if (g_nchildren < MAX_CHILDREN) g_children[g_nchildren++] = pid;
}

static int child_kill(int pid)
{
#ifdef _WIN32
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD)pid);
    if (!h) return -1;
    BOOL ok = TerminateProcess(h, 1);
    CloseHandle(h);
    return ok ? 0 : -1;
#else
    return kill(pid, SIGTERM);
#endif
}

/* 后台拉起 `bin gen --rank R ...`, 输出追加写入 logdir/rank<R>.log */
static int spawn_gen(const char* bin, const char* logdir,
                     int rank, int ranks, int port_base, int tokens,
                     const char* model, const char* vocab, const char* prompt)
{
    char log[1024];
    snprintf(log, sizeof(log), "%s/rank%d.log", logdir, rank);
#ifdef _WIN32
    char cmd[8192];
    snprintf(cmd, sizeof(cmd),
             "cmd.exe /c \"\"%s\" gen --model \"%s\" --vocab \"%s\" --tokens %d --prompt \"%s\" --rank %d --ranks %d --port-base %d >> \"%s\" 2>&1\"",
             bin, model, vocab, tokens, prompt, rank, ranks, port_base, log);
    STARTUPINFOA si; PROCESS_INFORMATION pi;
    memset(&si, 0, sizeof(si)); memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);
    if (!CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "worker: CreateProcess fail\n");
        return -1;
    }
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    child_add((int)pi.dwProcessId);
#else
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        setsid();
        char cmd[8192];
        snprintf(cmd, sizeof(cmd),
                 "\"%s\" gen --model \"%s\" --vocab \"%s\" --tokens %d --prompt \"%s\" --rank %d --ranks %d --port-base %d >> \"%s\" 2>&1",
                 bin, model, vocab, tokens, prompt, rank, ranks, port_base, log);
        execl("/bin/sh", "sh", "-c", cmd, (char*)NULL);
        _exit(127);
    }
    child_add((int)pid);
#endif
    return 0;
}

static int make_sock(void)
{
    return (int)socket(AF_INET, SOCK_STREAM, 0);
}

/* ---- 通用 socket 读写 ---- */

static int tcp_connect(const char* host, uint16_t port)
{
    int fd = make_sock();
    if (fd < 0) return -1;
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
#ifdef _WIN32
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1)
        addr.sin_addr.S_un.S_addr = inet_addr(host);
#else
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1)
        addr.sin_addr.s_addr = inet_addr(host);
#endif
    int attempt;
    for (attempt = 0; attempt < 50; attempt++) {
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) return fd;
#ifdef _WIN32
        Sleep(200);
#else
        struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 200 * 1000 * 1000;
        nanosleep(&ts, NULL);
#endif
    }
    close(fd);
    return -1;
}

static int xrecv(int fd, void* buf, size_t n)
{
    char* p = (char*)buf;
    while (n > 0) {
        size_t got;
#ifdef _WIN32
        int r = recv(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
        if (r <= 0) return -1;
        got = (size_t)r;
#else
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return -1;
        got = (size_t)r;
#endif
        p += got;
        n -= got;
    }
    return 0;
}

static void xsend(int fd, const void* buf, size_t n)
{
    const char* p = (const char*)buf;
    while (n > 0) {
        size_t sent;
#ifdef _WIN32
        int r = send(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
        if (r <= 0) return;
        sent = (size_t)r;
#else
        ssize_t r = send(fd, p, n, 0);
        if (r <= 0) return;
        sent = (size_t)r;
#endif
        p += sent;
        n -= sent;
    }
}

/* 读一行到 max(不含换行), 返回长度或 -1 */
static int recv_line(int fd, char* buf, size_t max)
{
    size_t n = 0;
    for (;;) {
        char c;
        ssize_t r;
#ifdef _WIN32
        r = (ssize_t)recv(fd, &c, 1, 0);
#else
        r = recv(fd, &c, 1, 0);
#endif
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n < max - 1) buf[n++] = c;
    }
    buf[n] = '\0';
    return (int)n;
}

/* ---- 本文件服务(私有 TCP 帧, 供 sync 拉模型) ---- */
/* 协议: 请求一行 `HEAD <rel>` 或 `GET <rel>`
 *      响应一行 `OK <size> <mtime>`; GET 的 OK 行后紧跟 size 字节文件内容
 * 错误: `ERR <msg>`  */

static long long file_info(const char* path, long long* mtime)
{
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    if (mtime) *mtime = (long long)st.st_mtime;
    return (long long)st.st_size;
}

static void serve_conn(int fd, const char* root)
{
    char line[MAX_LINE];
    if (recv_line(fd, line, sizeof(line)) < 0) { close(fd); return; }
    char op[16];
    char path[1024];
    if (sscanf(line, "%15s %1023s", op, path) != 2) { close(fd); return; }
    if (strcmp(op, "HEAD") != 0 && strcmp(op, "GET") != 0) {
        const char* e = "ERR bad op";
        xsend(fd, e, strlen(e));
        close(fd);
        return;
    }
    char full[4096];
    size_t rootlen = strlen(root);
    int need_slash = rootlen > 0 && root[rootlen - 1] != '/' && root[rootlen - 1] != '\\';
    if (snprintf(full, sizeof(full), "%s%s%s", root, need_slash ? "/" : "", path) >= (int)sizeof(full)) { close(fd); return; }
    long long mtime = 0;
    long long size = file_info(full, &mtime);
    if (size < 0) {
        const char* nf = "ERR not found";
        xsend(fd, nf, strlen(nf));
        close(fd);
        return;
    }
    char ok[128];
    int okn = snprintf(ok, sizeof(ok), "OK %lld %lld\n", size, mtime);
    xsend(fd, ok, (size_t)okn);
    if (strcmp(op, "GET") == 0) {
        FILE* f = fopen(full, "rb");
        if (f) {
            char buf[65536];
            long long left = size;
            while (left > 0) {
                size_t want = left < (long long)sizeof(buf) ? (size_t)left : sizeof(buf);
                size_t got = fread(buf, 1, want, f);
                if (got == 0) break;
                xsend(fd, buf, got);
                left -= (long long)got;
            }
            fclose(f);
        }
    }
    close(fd);
}

static int run_serve(uint16_t port, const char* root)
{
#ifdef _WIN32
    ws_init();
#endif
    int srv = make_sock();
    if (srv < 0) return 1;
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(srv); return 1; }
    if (listen(srv, 8) != 0) { close(srv); return 1; }
    fprintf(stderr, "worker: serving root %s on port %u\n", root, port);
    fflush(stderr);
    for (;;) {
        int fd = (int)accept(srv, NULL, NULL);
        if (fd < 0) continue;
        serve_conn(fd, root);
    }
    return 0;
}

/* ---- sync: 从 HOST:PORT 拉 REL 到本地 DEST ---- */
/* 先 HEAD 拿 size/mtime, 本地一致则跳过, 否则 GET 下载并写文件 */

static int sync_file(const char* host, uint16_t port, const char* rel, const char* dest)
{
    char line[MAX_LINE];
    int fd = tcp_connect(host, port);
    if (fd < 0) return -1;
    snprintf(line, sizeof(line), "HEAD %s\n", rel);
    xsend(fd, line, strlen(line));
    if (recv_line(fd, line, sizeof(line)) < 0) { close(fd); return -2; }
    long long rsize = -1, rmtime = -1;
    if (sscanf(line, "OK %lld %lld", &rsize, &rmtime) != 2) { close(fd); return -2; }
    close(fd);

    long long lsize = -1, lmtime = -1;
    struct stat st;
    if (stat(dest, &st) == 0) { lsize = (long long)st.st_size; lmtime = (long long)st.st_mtime; }
    /* 模型较大, 优先 mtime, 但 size 也要匹配: 两者一致才算最新;
     * 远端 mtime 缺失时才只退化为 size 比较 */
    if (rmtime >= 0) {
        if (lmtime == rmtime && lsize == rsize) return 0;
    } else {
        if (lsize == rsize) return 0;
    }

    /* 下载 */
    fd = tcp_connect(host, port);
    if (fd < 0) return -1;
    snprintf(line, sizeof(line), "GET %s\n", rel);
    xsend(fd, line, strlen(line));
    if (recv_line(fd, line, sizeof(line)) < 0) { close(fd); return -2; }
    long long gsize = -1, gmtime = -1;
    if (sscanf(line, "OK %lld %lld", &gsize, &gmtime) != 2) { close(fd); return -2; }
    if (gmtime >= 0 && rmtime < 0) rmtime = gmtime;

    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s.tmp", dest);
    FILE* f = fopen(tmp, "wb");
    if (!f) { DBG("sync: fopen tmp fail %s", tmp); close(fd); return -1; }
    char buf[65536];
    long long left = gsize;
    int ok = 1;
    while (left > 0 && ok) {
        size_t want = left < (long long)sizeof(buf) ? (size_t)left : sizeof(buf);
        if (xrecv(fd, buf, want) != 0) { ok = 0; break; }
        fwrite(buf, 1, want, f);
        left -= (long long)want;
    }
    close(fd);
    fclose(f);
    if (!ok || left != 0) { DBG("sync: short read left=%lld", left); remove(tmp); return -1; }
#ifdef _WIN32
    /* Windows rename 不会覆盖已存在文件, 先删目标再改名 */
    remove(dest);
#endif
    if (rename(tmp, dest) != 0) { DBG("sync: rename fail %s", strerror(errno)); remove(tmp); return -1; }
    struct utimbuf ub;
    ub.actime = (time_t)rmtime;
    ub.modtime = (time_t)rmtime;
    utime(dest, &ub);
    return 1; /* 已下载 */
}

/* ---- worker 协议命令 ---- */

static void handle_line(char* line, const char* bin, const char* logdir,
                        char* out, size_t outsz)
{
    out[0] = '\0';
    char* cmd = line;
    if (strcmp(cmd, "ping") == 0) { snprintf(out, outsz, "ok"); return; }
    if (strcmp(cmd, "quit") == 0) { snprintf(out, outsz, "ok"); return; }
    if (strcmp(cmd, "stop") == 0) {
        int i;
        for (i = 0; i < g_nchildren; i++) child_kill(g_children[i]);
        g_nchildren = 0;
        snprintf(out, outsz, "ok");
        return;
    }
    if (strncmp(cmd, "sync ", 5) == 0) {
        char host[128], rel[1024], dest[1024];
        int port = 0;
        if (sscanf(cmd + 5, "%127s %d %1023s %1023s", host, &port, rel, dest) == 4) {
            int r = sync_file(host, (uint16_t)port, rel, dest);
            if (r == 0) snprintf(out, outsz, "ok up-to-date %s", dest);
            else if (r == 1) snprintf(out, outsz, "ok downloaded %s", dest);
            else snprintf(out, outsz, "err sync(%d) %s", r, dest);
            return;
        }
        snprintf(out, outsz, "err bad sync args");
        return;
    }
    if (strncmp(cmd, "run", 3) == 0 && (cmd[3] == ' ')) {
        char* p = cmd + 4;
        char* tok[7];
        int i;
        for (i = 0; i < 6; i++) {
            tok[i] = strtok(p, " ");
            if (!tok[i]) break;
            p = NULL;
        }
        /* prompt = 第6个字段之后的行尾剩余部分(可含空格) */
        if (i >= 6) {
            char* rest = strtok(NULL, "");
            if (tok[0] && tok[1] && tok[2] && tok[3] && tok[4] && tok[5] && rest) {
                int rank = atoi(tok[0]);
                int ranks = atoi(tok[1]);
                int port_base = atoi(tok[2]);
                int tokens = atoi(tok[3]);
                if (spawn_gen(bin, logdir, rank, ranks, port_base, tokens,
                              tok[4], tok[5], rest) == 0)
                    snprintf(out, outsz, "ok spawned rank %d", rank);
                else
                    snprintf(out, outsz, "err spawn");
                return;
            }
        }
        snprintf(out, outsz, "err bad args");
        return;
    }
    snprintf(out, outsz, "err unknown cmd");
}

static int run_server(uint16_t port, const char* bin, const char* logdir)
{
#ifdef _WIN32
    ws_init();
#endif
    int srv = make_sock();
    if (srv < 0) return 1;
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0) { close(srv); return 1; }
    if (listen(srv, 8) != 0) { close(srv); return 1; }
    fprintf(stderr, "worker: listening on port %u (bin=%s logdir=%s)\n", port, bin, logdir);
    fflush(stderr);

for (;;) {
        int fd = (int)accept(srv, NULL, NULL);
        if (fd < 0) continue;
        DBG("worker: accepted fd=%d", fd);
        char line[MAX_LINE];
        if (recv_line(fd, line, sizeof(line)) < 0) {
#ifdef _WIN32
            DBG("worker: recv_line fail fd=%d WSAErr=%d", fd, WSAGetLastError());
#else
            DBG("worker: recv_line fail fd=%d", fd);
#endif
            close(fd); continue;
        }
        DBG("worker: got cmd [%s]", line);
        char out[64];
        handle_line(line, bin, logdir, out, sizeof(out));
        DBG("worker: reply [%s]", out);
        char reply[MAX_LINE];
        snprintf(reply, sizeof(reply), "%s\n", out);
        xsend(fd, reply, strlen(reply));
        DBG("worker: sent reply, closing");
        close(fd);
        if (strcmp(out, "ok") == 0 && strcmp(line, "quit") == 0) break;
    }
    close(srv);
    return 0;
}

static int run_client(const char* host, uint16_t port, const char* send)
{
#ifdef _WIN32
    ws_init();
#endif
    int fd = tcp_connect(host, port);
    if (fd < 0) { fprintf(stderr, "connect fail\n"); return 1; }
    char line[MAX_LINE];
    snprintf(line, sizeof(line), "%s\n", send);
    xsend(fd, line, strlen(line));
    char out[MAX_LINE];
    if (recv_line(fd, out, sizeof(out)) >= 0) {
        printf("%s\n", out);
#ifdef _WIN32
        Sleep(300);
#else
        usleep(300 * 1000);
#endif
    }
    close(fd);
    return 0;
}

int main(int argc, char** argv)
{
    const char* host = "127.0.0.1";
    const char* bin = "./build/avx2/yllm";
    const char* logdir = "./logs";
    const char* root = NULL;
    const char* send = NULL;
    int serve_mode = 0;
    uint16_t port = 9100;
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) host = argv[++i];
        else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc) bin = argv[++i];
        else if (strcmp(argv[i], "--logdir") == 0 && i + 1 < argc) logdir = argv[++i];
        else if (strcmp(argv[i], "--serve") == 0) serve_mode = 1;
        else if (strcmp(argv[i], "--root") == 0 && i + 1 < argc) root = argv[++i];
        else if (strcmp(argv[i], "--send") == 0 && i + 1 < argc) send = argv[++i];
        else { fprintf(stderr, "usage: dist-worker [--host ip] [--port n] [--bin path] [--logdir dir] [--send cmd] | --serve [--root dir]\n"); return 1; }
    }
    if (serve_mode) {
        if (!root) root = ".";
        return run_serve(port, root);
    }
    if (send) return run_client(host, port, send);
    return run_server(port, bin, logdir);
}