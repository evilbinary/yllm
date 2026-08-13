/* http.h — 最小 HTTP/1.1 服务器(OpenAI 兼容层用)
 *
 * 支持: 请求行/头部/body 解析; 响应(JSON); SSE 流式。
 * 单线程逐连接处理(推理服务场景, 每请求一个连接)。
 */

#ifndef YLLM_SERVE_HTTP_H
#define YLLM_SERVE_HTTP_H

#include "sock.h"
#include <stddef.h>
#ifdef _WIN32
#include <string.h>
#define strncasecmp _strnicmp
#else
#include <strings.h>
#endif

#define HTTP_MAX_LINE 8192
#define HTTP_MAX_HEADERS 32

typedef struct {
    char method[16];
    char path[512];
    char version[16];
    char headers[HTTP_MAX_HEADERS][512];
    int n_headers;
    char* body;          /* 请求 body(解析时分配, 调用方负责释放) */
    size_t body_len;
} HttpRequest;

typedef struct {
    int fd;
    int status;
    const char* content_type;
    int chunked;         /* SSE 用: 连接保持, 流式写 */
} HttpResponse;

/* 解析请求(阻塞读请求行+头+body)。返回 0 成功, -1 失败 */
static inline int http_parse_request(int fd, HttpRequest* req)
{
    memset(req, 0, sizeof(*req));
    char line[HTTP_MAX_LINE];
    int n = sock_recv_line(fd, line, sizeof(line));
    if (n < 0) return -1;
    if (sscanf(line, "%15s %511s %15s", req->method, req->path, req->version) != 3)
        return -1;
    /* 请求头 */
    while (req->n_headers < HTTP_MAX_HEADERS) {
        n = sock_recv_line(fd, line, sizeof(line));
        if (n < 0) return -1;
        if (n == 0) break;   /* 空行 */
        snprintf(req->headers[req->n_headers], sizeof(req->headers[req->n_headers]), "%s", line);
        req->n_headers++;
    }
    /* body */
    size_t clen = 0;
    int i;
    for (i = 0; i < req->n_headers; i++) {
        if (strncasecmp(req->headers[i], "Content-Length:", 15) == 0) {
            clen = (size_t)atol(req->headers[i] + 15);
        }
    }
    if (clen > 0) {
        req->body = (char*)malloc(clen + 1);
        if (!req->body) return -1;
        if (sock_recv_n(fd, req->body, clen) != 0) {
            free(req->body);
            req->body = NULL;
            return -1;
        }
        req->body[clen] = '\0';
        req->body_len = clen;
    }
    return 0;
}

/* 开始响应 */
static inline void http_begin(HttpResponse* r, int fd, int status, const char* content_type)
{
    memset(r, 0, sizeof(*r));
    r->fd = fd;
    r->status = status;
    r->content_type = content_type ? content_type : "application/json";
}

static inline void http_write_head(HttpResponse* r)
{
    char buf[512];
    int n = snprintf(buf, sizeof(buf),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     r->status,
                     r->status == 200 ? "OK" :
                     r->status == 404 ? "Not Found" :
                     r->status == 400 ? "Bad Request" :
                     r->status == 503 ? "Service Unavailable" : "Error",
                     r->content_type);
    sock_send_n(r->fd, buf, (size_t)n);
}

/* 写完整响应(body 一次给全) */
static inline void http_reply(HttpResponse* r, const char* body)
{
    http_write_head(r);
    sock_send_n(r->fd, body, strlen(body));
}

/* SSE: 开始流(保持连接, 分块写) */
static inline void http_sse_begin(HttpResponse* r, int fd)
{
    memset(r, 0, sizeof(*r));
    r->fd = fd;
    r->status = 200;
    r->content_type = "text/event-stream";
    http_write_head(r);
}

/* SSE: 写一个 data 块 */
static inline void http_sse_data(HttpResponse* r, const char* data)
{
    sock_send_n(r->fd, "data: ", 6);
    sock_send_n(r->fd, data, strlen(data));
    sock_send_n(r->fd, "\n\n", 2);
}

/* SSE: 结束 */
static inline void http_sse_done(HttpResponse* r)
{
    sock_send_n(r->fd, "data: [DONE]\n\n", 14);
}

#endif /* YLLM_SERVE_HTTP_H */
