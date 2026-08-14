/* sock.h — serve 层公共 socket 辅助(rank / server / router 共用) */

#ifndef YLLM_SERVE_SOCK_H
#define YLLM_SERVE_SOCK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#define ssize_t int
static inline int sock_close(int fd) { return closesocket((SOCKET)fd); }
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <fcntl.h>
#include <arpa/inet.h>
static inline int sock_close(int fd) { return close(fd); }
#endif

static inline void sock_init(void)
{
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN); /* 对端关闭时不自杀, send 返回 EPIPE 由上层处理 */
#else
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
}

static inline int sock_recv_n(int fd, void* buf, size_t n)
{
    char* p = (char*)buf;
    while (n > 0) {
#ifdef _WIN32
        int r = recv(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
#else
        ssize_t r = recv(fd, p, n, 0);
#endif
        if (r <= 0) return -1;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 0;
}

static inline void sock_send_n(int fd, const void* buf, size_t n)
{
    const char* p = (const char*)buf;
    while (n > 0) {
#ifdef _WIN32
        int r = send(fd, p, (int)(n > 0x7fffffff ? 0x7fffffff : n), 0);
#else
        ssize_t r = send(fd, p, n, 0);
#endif
        if (r <= 0) return;
        p += (size_t)r;
        n -= (size_t)r;
    }
}

/* 读一行到 max(不含换行), 返回长度或 -1 */
static inline int sock_recv_line(int fd, char* buf, size_t max)
{
    size_t n = 0;
    for (;;) {
        char c;
#ifdef _WIN32
        int r = recv(fd, &c, 1, 0);
#else
        ssize_t r = recv(fd, &c, 1, 0);
#endif
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n < max - 1) buf[n++] = c;
    }
    buf[n] = '\0';
    return (int)n;
}

/* 发送一行文本 */
static inline void sock_send_line(int fd, const char* fmt, ...)
{
    char buf[8192];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    sock_send_n(fd, buf, strlen(buf));
    sock_send_n(fd, "\n", 1);
}

/* 连接后设置接收超时(秒); 超时后 recv/sock_recv_line 返回 -1 */
static inline void sock_set_timeout(int fd, int sec)
{
    struct timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));
}

/* TCP 连接(带重试) */
static inline int sock_connect(const char* host, uint16_t port, int retries)
{
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
    for (attempt = 0; attempt < retries; attempt++) {
        int fd = (int)socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
#ifndef _WIN32
        fcntl(fd, F_SETFD, FD_CLOEXEC); /* fork+exec 时不泄漏 fd 给子进程 */
#endif
        if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            int one = 1;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
            return fd;
        }
        sock_close(fd);
#ifdef _WIN32
        Sleep(200);
#else
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 200 * 1000 * 1000;
        nanosleep(&ts, NULL);
#endif
    }
    return -1;
}

/* TCP 连接: host 为子串(长度已知), 用于 "ip:port" 字符串直接连接 */
static inline int sock_connect_host(const char* host, size_t hlen, uint16_t port, int retries)
{
    char tmp[128];
    if (hlen >= sizeof(tmp)) hlen = sizeof(tmp) - 1;
    memcpy(tmp, host, hlen);
    tmp[hlen] = '\0';
    return sock_connect(tmp, port, retries);
}

/* 跨平台睡眠(毫秒) */
static inline void sock_sleep_ms(int ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000 * 1000;
    nanosleep(&ts, NULL);
#endif
}

/* 创建 TCP 监听 socket(INADDR_ANY), 成功返回 fd, 失败返回 -1 */
static inline int sock_listen(uint16_t port, int backlog)
{
    int srv = (int)socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) return -1;
#ifndef _WIN32
    fcntl(srv, F_SETFD, FD_CLOEXEC); /* fork+exec 时不泄漏 fd 给子进程 */
#endif
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0) { sock_close(srv); return -1; }
    if (listen(srv, backlog) != 0) { sock_close(srv); return -1; }
    return srv;
}

/* 等待可读(毫秒超时), 有连接返回 accept 的 fd, 超时返回 -2, 错误返回 -1。
 * 用于所有节点主循环: 非阻塞 accept + 周期任务(心跳/判死/查询)。 */
static inline int sock_accept_with_timeout(int srv, int ms)
{
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(srv, &rfds);
    int sel = select(srv + 1, &rfds, NULL, NULL, &tv);
    if (sel <= 0) return sel == 0 ? -2 : -1;
    if (!FD_ISSET(srv, &rfds)) return -2;
    return (int)accept(srv, NULL, NULL);
}

/* 等待 fd 可读(毫秒超时): 可读返回 1, 超时返回 0, 错误返回 -1。
 * 用于收流前等待对端数据(避免阻塞 recv 的边界行为)。 */
static inline int sock_wait_readable(int fd, int ms)
{
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    int sel = select(fd + 1, &rfds, NULL, NULL, &tv);
    if (sel < 0) return -1;
    return sel > 0 && FD_ISSET(fd, &rfds) ? 1 : 0;
}

#endif /* YLLM_SERVE_SOCK_H */
