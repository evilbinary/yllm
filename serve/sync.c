/* sync.c — 文件分发(设计 §3.5 supervisor 文件分发)
 *   接收端: yllm sync --serve --port <N> [--dir <dest-dir>]
 *   发送端: yllm sync --push <local-file> --to <host:port> --dest <remote-path>
 *
 * 帧: FILE_PUT <path> <size>\n + <size bytes> → 写文件 → 回 OK
 */
#include "frame.h"
#include "sock.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static int recv_file(int fd, const char* args, const char* dir)
{
    char path[1024];
    long size = 0;
    if (sscanf(args, "%1023s %ld", path, &size) != 2 || size < 0) {
        frame_send(fd, "ERR", "bad FILE_PUT args");
        return 1;
    }
    /* 防目录穿越 */
    char full[2048];
    snprintf(full, sizeof(full), "%s/%s", dir, path);
    const char* p = full;
    for (; *p; p++) if (p[0] == '.' && p[1] == '.') {
        frame_send(fd, "ERR", "path traversal rejected");
        return 1;
    }
    char* slash = strrchr(full, '/');
    if (slash) {
        *slash = 0;
#ifndef _WIN32
        mkdir(full, 0755);
#else
        _mkdir(full);
#endif
        *slash = '/';
    }
    FILE* f = fopen(full, "wb");
    if (!f) { frame_send(fd, "ERR", "cannot create dest file"); return 1; }
    char buf[65536];
    long left = size;
    while (left > 0) {
        size_t want = (size_t)(left > (long)sizeof(buf) ? (long)sizeof(buf) : left);
        if (sock_recv_n(fd, buf, want) != 0) { fclose(f); return 1; }
        fwrite(buf, 1, want, f);
        left -= (long)want;
    }
    fclose(f);
    frame_send(fd, "OK", path);
    printf("sync: received %s (%ld bytes)\n", full, size);
    return 0;
}

static int serve_sync(int port, const char* dir)
{
    sock_init();
    int srv = sock_listen((uint16_t)port, 8);
    if (srv < 0) { fprintf(stderr, "sync: cannot listen on %d\n", port); return 1; }
    printf("sync: listening on port %d (dir=%s)\n", port, dir);
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd < 0) continue;
        Frame f;
        if (frame_recv(fd, &f) >= 0) {
            if (strcmp(f.cmd, "FILE_PUT") == 0) recv_file(fd, f.args, dir);
            else if (strcmp(f.cmd, "QUIT") == 0) { frame_send(fd, "OK", NULL); close(fd); break; }
            else frame_send(fd, "ERR", "unknown cmd");
        }
        close(fd);
    }
    close(srv);
    return 0;
}

static int push_file(const char* file, const char* to, const char* dest)
{
    sock_init();
    char host[128];
    int port = 0;
    {
        const char* colon = strchr(to, ':');
        if (!colon) { fprintf(stderr, "sync: bad --to <host:port>\n"); return 1; }
        size_t hlen = (size_t)(colon - to);
        if (hlen >= sizeof(host)) hlen = sizeof(host) - 1;
        memcpy(host, to, hlen);
        host[hlen] = 0;
        port = atoi(colon + 1);
    }
    FILE* f = fopen(file, "rb");
    if (!f) { fprintf(stderr, "sync: cannot open %s\n", file); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    int fd = sock_connect(host, (uint16_t)port, 3);
    if (fd < 0) { fclose(f); fprintf(stderr, "sync: cannot connect %s\n", to); return 1; }
    char args[1100];
    snprintf(args, sizeof(args), "%s %ld", dest, size);
    frame_send(fd, "FILE_PUT", args);
    char buf[65536];
    long left = size;
    while (left > 0) {
        size_t want = (size_t)(left > (long)sizeof(buf) ? (long)sizeof(buf) : left);
        size_t n = fread(buf, 1, want, f);
        if (n == 0) break;
        sock_send_n(fd, buf, n);
        left -= (long)n;
    }
    fclose(f);
    Frame r;
    if (frame_recv(fd, &r) >= 0) printf("sync: %s %s\n", r.cmd, r.args);
    close(fd);
    return 0;
}

int cmd_sync(int argc, char** argv)
{
    const char* mode = NULL;
    int port = 0;
    const char* dir = ".";
    const char* file = NULL;
    const char* to = NULL;
    const char* dest = NULL;
    int i;
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--serve") == 0) mode = "serve";
        else if (strcmp(argv[i], "--push") == 0) { mode = "push"; file = argv[++i]; }
        else if (strcmp(argv[i], "--port") == 0) port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dir") == 0) dir = argv[++i];
        else if (strcmp(argv[i], "--to") == 0) to = argv[++i];
        else if (strcmp(argv[i], "--dest") == 0) dest = argv[++i];
    }
    if (!mode || (strcmp(mode, "serve") == 0 && port <= 0) ||
        (strcmp(mode, "push") == 0 && (!file || !to || !dest))) {
        fprintf(stderr, "usage: yllm sync --serve --port <N> [--dir <dir>]\n");
        fprintf(stderr, "   or: yllm sync --push <file> --to <host:port> --dest <remote-path>\n");
        return 1;
    }
    if (strcmp(mode, "serve") == 0) return serve_sync(port, dir);
    return push_file(file, to, dest);
}
