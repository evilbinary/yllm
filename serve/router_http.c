/* router_http.c — OpenAI 兼容 HTTP 层(router 的对外入口)
 *
 * 端点:
 *   POST /v1/chat/completions    chat, 支持 stream(SSE)与 JSON
 *   POST /v1/completions         文本补全
 *   GET  /v1/models              已注册模型列表
 *   GET  /health                 存活/就绪
 *
 * 由 router 的 --http-port 启用, 独立线程跑, 复用 Router 路由逻辑。
 */
#include "router.h"
#include "json.h"
#include "http.h"
#include "../inference/yllm.h"
#include "../inference/log.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

#define HTTP_MAX_BODY (1 << 20)

/* API 请求/响应日志开关(默认开启; hub/router 按 --api-log 配置) */
static int g_api_log = 1;
void router_http_set_api_log(int on) { g_api_log = on; }

typedef struct {
    Router* router;
    uint16_t port;
} HttpCtx;

/* 非流式收集回调上下文 */
typedef struct {
    char* buf;
    size_t cap;
    size_t len;
    int n_tokens;        /* 收集到的 token 数(usage 统计用) */
} CollectCtx;

/* 收集回调: token 拼进 buffer(转义 JSON 特殊字符) */
static void collect_on_token(const char* utf8, size_t len, void* ctx)
{
    CollectCtx* cc = (CollectCtx*)ctx;
    size_t i;
    for (i = 0; i < len && cc->len + 1 < cc->cap; i++) {
        char c = utf8[i];
        if (c == '"' || c == '\\') {
            if (cc->len + 2 < cc->cap) { cc->buf[cc->len++] = '\\'; cc->buf[cc->len++] = c; }
        } else if (c == '\n') {
            if (cc->len + 2 < cc->cap) { cc->buf[cc->len++] = '\\'; cc->buf[cc->len++] = 'n'; }
        } else if (c == '\r') {
            if (cc->len + 2 < cc->cap) { cc->buf[cc->len++] = '\\'; cc->buf[cc->len++] = 'r'; }
        } else if (c == '\t') {
            if (cc->len + 2 < cc->cap) { cc->buf[cc->len++] = '\\'; cc->buf[cc->len++] = 't'; }
        } else {
            cc->buf[cc->len++] = c;
        }
    }
    cc->buf[cc->len] = '\0';
    cc->n_tokens++;
}

/* 完整打印请求 body 到日志(绕过 4096 日志缓冲截断, 分块写) */
static void ylog_body(const char* tag, const char* body)
{
    if (!g_api_log) return;
    if (!body) { ylog_info("%s body=NULL", tag); return; }
    size_t blen = strlen(body);
    ylog_info("%s body_len=%zu", tag, blen);
    const size_t CHUNK = 2000;
    size_t off = 0;
    while (off < blen) {
        size_t take = (blen - off > CHUNK) ? CHUNK : (blen - off);
        char buf[CHUNK + 1];
        memcpy(buf, body + off, take);
        buf[take] = '\0';
        ylog_raw("%s", buf);
        ylog_raw("\n");
        off += take;
    }
}

/* SSE 回调: 逐 token 写 data 块 */
typedef struct {
    HttpResponse* r;
    const char* model;
    int n_tokens;        /* 已发出的内容 token 数(usage 统计) */
    int include_usage;   /* 客户端请求 stream_options.include_usage */
} SseCtx;

static void sse_on_token(const char* utf8, size_t len, void* ctx)
{
    SseCtx* sc = (SseCtx*)ctx;
    HttpResponse* r = sc->r;
    sc->n_tokens++;
    char* escaped = (char*)malloc(len * 2 + 1);
    if (!escaped) return;
    size_t o = 0, i;
    for (i = 0; i < len; i++) {
        if (utf8[i] == '"') { escaped[o++] = '\\'; escaped[o++] = '"'; }
        else if (utf8[i] == '\\') { escaped[o++] = '\\'; escaped[o++] = '\\'; }
        else if (utf8[i] == '\n') { escaped[o++] = '\\'; escaped[o++] = 'n'; }
        else if (utf8[i] == '\r') { escaped[o++] = '\\'; escaped[o++] = 'r'; }
        else if (utf8[i] == '\t') { escaped[o++] = '\\'; escaped[o++] = 't'; }
        else escaped[o++] = utf8[i];
    }
    escaped[o] = '\0';
    size_t jlen = 256 + len * 2;
    char* json = (char*)malloc(jlen);
    if (!json) { free(escaped); return; }
    snprintf(json, jlen,
             "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
             (long long)time(NULL), sc->model, escaped);
    http_sse_data(r, json);
    free(json);
    free(escaped);
}

/* SSE 收尾 chunk: OpenAI 要求流结束时必须带 finish_reason(否则客户端报 "Stream ended without finish_reason") */
static void sse_on_done(SseCtx* sc, const char* reason)
{
    char json[512];
    snprintf(json, sizeof(json),
             "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"%s\","
             "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}",
             (long long)time(NULL), sc->model, reason);
    http_sse_data(sc->r, json);
    /* include_usage 请求时补 usage 收尾 chunk(空 choices), 否则客户端 tok/s 统计为 0 */
    if (sc->include_usage) {
        int ct = sc->n_tokens;
        snprintf(json, sizeof(json),
                 "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[],\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                 (long long)time(NULL), sc->model, ct, ct);
        http_sse_data(sc->r, json);
    }
}

/* 会话 key: 客户端 IP + 首条用户消息哈希(同一对话稳定) */
static void make_sess_key(int fd, const char* first, size_t flen, char* out, size_t outsz)
{
    char ip[64] = "0.0.0.0";
    struct sockaddr_in sa;
    socklen_t sl = sizeof(sa);
    if (getpeername(fd, (struct sockaddr*)&sa, &sl) == 0)
        inet_ntop(AF_INET, &sa.sin_addr, ip, sizeof(ip));
    uint64_t h = 1469598103934665603ull;
    size_t i;
    for (i = 0; i < flen; i++) { h ^= (unsigned char)first[i]; h *= 1099511628211ull; }
    snprintf(out, outsz, "%s:%llx", ip, (unsigned long long)h);
}

/* 提取 chat 请求的全部消息(role/content), 返回消息数 */
static int extract_chat_messages(const char* body, char* pool, size_t poolsz,
                                 char** roles, char** contents, int max_msgs)
{
    int n = 0, i;
    size_t off = 0;
    for (i = 0; i < 32 && n < max_msgs; i++) {
        char path[64];
        JsonVal v;
        roles[n] = NULL;
        contents[n] = NULL;
        snprintf(path, sizeof(path), "messages.%d.role", i);
        if (json_find(body, path, &v) && v.type == JSON_STR && off + 64 < poolsz) {
            size_t rlen = json_str_unescape(v.start, v.len, pool + off);
            roles[n] = pool + off;
            off += rlen + 1;
        }
        if (!roles[n]) break;   /* 没有更多消息 */
        snprintf(path, sizeof(path), "messages.%d.content", i);
        if (json_find(body, path, &v) && v.type == JSON_STR && off + 64 < poolsz) {
            contents[n] = pool + off;
            off += json_str_unescape(v.start, v.len, pool + off) + 1;
        } else {
            contents[n] = pool + off;
            pool[off++] = '\0';
        }
        n++;
    }
    return n;
}

/* 提取 chat 请求的最后一条 content 作为 prompt */
static size_t extract_chat_prompt(const char* body, char* out, size_t outsz)
{
    JsonVal v;
    /* 尝试 messages.<i>.content, 从后往前(最后一条消息) */
    int i;
    for (i = 31; i >= 0; i--) {
        char path[64];
        snprintf(path, sizeof(path), "messages.%d.content", i);
        if (json_find(body, path, &v) && v.type == JSON_STR && v.len < outsz) {
            return json_str_unescape(v.start, v.len, out);
        }
    }
    out[0] = '\0';
    return 0;
}

/* ---- /v1/chat/completions ---- */
static void handle_chat_completions(int fd, Router* r, const char* body, int stream)
{
    JsonVal v;
    char model[128] = "default";
    if (json_find(body, "model", &v) && v.type == JSON_STR && v.len < sizeof(model)) {
        memcpy(model, v.start, v.len);
        model[v.len] = '\0';
    }
    /* 大 buffer 用堆(线程栈 1MB, 避免栈溢出) */
    char* prompt = (char*)malloc(HTTP_MAX_BODY);
    char* collected = (char*)malloc(HTTP_MAX_BODY);
    if (!prompt || !collected) {
        free(prompt);
        free(collected);
        HttpResponse rr;
        http_begin(&rr, fd, 500, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"oom\"}}");
        return;
    }
    size_t plen = extract_chat_prompt(body, prompt, HTTP_MAX_BODY);
    int max_tokens = 1024;
    if (json_find(body, "max_tokens", &v) && v.type == JSON_NUM)
        max_tokens = (int)json_num(&v);
    float rtemp = 1.0f, rtop_p = 0.9f;
    if (json_find(body, "temperature", &v) && v.type == JSON_NUM)
        rtemp = (float)json_num(&v);
    if (json_find(body, "top_p", &v) && v.type == JSON_NUM)
        rtop_p = (float)json_num(&v);
    int include_usage = 0;
    if (json_find(body, "stream_options.include_usage", &v) && v.type == JSON_BOOL) {
        /* true/false 均可能(JSON_BOOL 表示 true 或 false); 取值并判断 true */
        int bv = 0;
        if (v.start) bv = (v.start[0] == 't');
        include_usage = bv;
    }
    /* 会话模式: 提取全部消息 + 会话 key, 渲染后只发增量 token 给 rank */
    char* pool = (char*)malloc(HTTP_MAX_BODY);
    char* roles[16];
    char* contents[16];
    int n_msgs = 0;
    int sess_ok = 0;
    char sess_key[128] = "";
    if (pool) {
        n_msgs = extract_chat_messages(body, pool, HTTP_MAX_BODY, roles, contents, 16);
        if (n_msgs > 0) {
            make_sess_key(fd, contents[0], strlen(contents[0]), sess_key, sizeof(sess_key));
            sess_ok = 1;
        }
#if YLLM_SESS_DEBUG
        ylog_info("router_http: sess_ok=%d n_msgs=%d key=%s", sess_ok, n_msgs, sess_key);
#endif
    }

    if (plen == 0 && n_msgs == 0) {
        if (pool) free(pool);
        free(prompt);
        free(collected);
        HttpResponse rr;
        http_begin(&rr, fd, 400, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"empty prompt\"}}");
        return;
    }

    if (stream) {
        HttpResponse rr;
        SseCtx sc;
        sc.r = &rr;
        sc.model = model;
        sc.n_tokens = 0;
        sc.include_usage = include_usage;
        if (g_api_log)
            ylog_info("HTTP CHAT stream model=%s max_tokens=%d temp=%.3g top_p=%.3g include_usage=%d sess_ok=%d n_msgs=%d plen=%zu",
                      model, max_tokens, rtemp, rtop_p, include_usage, sess_ok, n_msgs, plen);
        http_sse_begin(&rr, fd);
        int rc;
        if (sess_ok)
            rc = router_infer_sess(r, model, max_tokens, sess_key,
                                   contents[n_msgs - 1], strlen(contents[n_msgs - 1]), sse_on_token, &sc,
                                   rtemp, rtop_p);
        else
            rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &sc,
                              rtemp, rtop_p);
        if (rc != 0) {
            if (rc == -2)
                http_sse_data(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            else
                http_sse_data(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
        } else {
            sse_on_done(&sc, "stop");
        }
        http_sse_done(&rr);
        if (g_api_log) ylog_info("HTTP CHAT stream done rc=%d n_tokens=%d", rc, sc.n_tokens);
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        int rc;
        if (sess_ok)
            rc = router_infer_sess(r, model, max_tokens, sess_key,
                                   contents[n_msgs - 1], strlen(contents[n_msgs - 1]), collect_on_token, &cc,
                                   rtemp, rtop_p);
        else
rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc,
                              rtemp, rtop_p);
        if (rc != 0) {
            HttpResponse rr;
            if (rc == -2) {
                http_begin(&rr, fd, 404, "application/json");
                http_reply(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            } else {
                http_begin(&rr, fd, 502, "application/json");
                http_reply(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
            }
            free(prompt);
            free(collected);
            return;
        }
        char* json = (char*)malloc(HTTP_MAX_BODY + 512);
        if (!json) { free(prompt); free(collected); return; }
        snprintf(json, HTTP_MAX_BODY + 512,
                 "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                 "\"finish_reason\":\"length\"}],"
                 "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                 (long long)time(NULL), model, collected, cc.n_tokens, cc.n_tokens);
        HttpResponse rr;
        http_begin(&rr, fd, 200, "application/json");
        http_reply(&rr, json);
        if (g_api_log) { ylog_info("HTTP CHAT reply n_tokens=%d", cc.n_tokens); ylog_body("HTTP CHAT reply-json", json); }
        free(json);
    }
    if (pool) free(pool);
    free(prompt);
    free(collected);
}

/* ---- /v1/completions ---- */
static void handle_completions(int fd, Router* r, const char* body, int stream)
{
    JsonVal v;
    char model[128] = "default";
    if (json_find(body, "model", &v) && v.type == JSON_STR && v.len < sizeof(model)) {
        memcpy(model, v.start, v.len);
        model[v.len] = '\0';
    }
    char* prompt = (char*)malloc(HTTP_MAX_BODY);
    char* collected = (char*)malloc(HTTP_MAX_BODY);
    if (!prompt || !collected) {
        free(prompt);
        free(collected);
        HttpResponse rr;
        http_begin(&rr, fd, 500, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"oom\"}}");
        return;
    }
    size_t plen = 0;
    if (json_find(body, "prompt", &v) && v.type == JSON_STR)
        plen = json_str_unescape(v.start, v.len, prompt);
    int max_tokens = 1024;
    if (json_find(body, "max_tokens", &v) && v.type == JSON_NUM)
        max_tokens = (int)json_num(&v);
    float rtemp = 1.0f, rtop_p = 0.9f;
    if (json_find(body, "temperature", &v) && v.type == JSON_NUM)
        rtemp = (float)json_num(&v);
    if (json_find(body, "top_p", &v) && v.type == JSON_NUM)
        rtop_p = (float)json_num(&v);
    int include_usage = 0;
    if (json_find(body, "stream_options.include_usage", &v) && v.type == JSON_BOOL) {
        int bv = 0;
        if (v.start) bv = (v.start[0] == 't');
        include_usage = bv;
    }

    if (plen == 0) {
        free(prompt);
        free(collected);
        HttpResponse rr;
        http_begin(&rr, fd, 400, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"empty prompt\"}}");
        return;
    }
    if (stream) {
        HttpResponse rr;
        SseCtx sc;
        sc.r = &rr;
        sc.model = model;
        sc.n_tokens = 0;
        sc.include_usage = include_usage;
        if (g_api_log)
            ylog_info("HTTP COMPLETIONS stream model=%s max_tokens=%d temp=%.3g top_p=%.3g include_usage=%d plen=%zu",
                      model, max_tokens, rtemp, rtop_p, include_usage, plen);
        http_sse_begin(&rr, fd);
        int rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &sc,
                              rtemp, rtop_p);
        if (rc != 0) {
            if (rc == -2)
                http_sse_data(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            else
                http_sse_data(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
        } else {
            sse_on_done(&sc, "length");
        }
        http_sse_done(&rr);
        if (g_api_log) ylog_info("HTTP COMPLETIONS stream done rc=%d n_tokens=%d", rc, sc.n_tokens);
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        int rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc,
                              rtemp, rtop_p);
        if (rc != 0) {
            HttpResponse rr;
            if (rc == -2) {
                http_begin(&rr, fd, 404, "application/json");
                http_reply(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            } else {
                http_begin(&rr, fd, 502, "application/json");
                http_reply(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
            }
            free(prompt);
            free(collected);
            return;
        }
        char* json = (char*)malloc(HTTP_MAX_BODY + 256);
        if (!json) { free(prompt); free(collected); return; }
        snprintf(json, HTTP_MAX_BODY + 256,
                 "{\"id\":\"cmpl-1\",\"object\":\"text_completion\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[{\"text\":\"%s\",\"index\":0,\"finish_reason\":\"length\"}]}",
                 (long long)time(NULL), model, collected);
        HttpResponse rr;
        http_begin(&rr, fd, 200, "application/json");
        http_reply(&rr, json);
        if (g_api_log) { ylog_info("HTTP COMPLETIONS reply n_tokens=%d", cc.n_tokens); ylog_body("HTTP COMPLETIONS reply-json", json); }
        free(json);
    }
    free(prompt);
    free(collected);
}

/* ---- /v1/models ---- */
static void handle_models(int fd, Router* r)
{
    char json[2048] = "{\"object\":\"list\",\"data\":[";
    int first = 1;
    int i;
    pthread_mutex_lock(&r->lock);
    for (i = 0; i < r->n_servers; i++) {
        if (r->servers[i].state != 1) continue;
        if (!first) strncat(json, ",", sizeof(json) - strlen(json) - 1);
        first = 0;
        char one[256];
        snprintf(one, sizeof(one), "{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"yllm\"}",
                 r->servers[i].model);
        strncat(json, one, sizeof(json) - strlen(json) - 1);
    }
    pthread_mutex_unlock(&r->lock);
    strncat(json, "]}", sizeof(json) - strlen(json) - 1);
    HttpResponse rr;
    http_begin(&rr, fd, 200, "application/json");
    http_reply(&rr, json);
}

static void handle_conn(int fd, Router* r)
{
    HttpRequest req;
    if (http_parse_request(fd, &req) != 0) {
        sock_close(fd);
        return;
    }

    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        HttpResponse rr;
        http_begin(&rr, fd, 200, "application/json");
        http_reply(&rr, "{\"status\":\"ok\"}");
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(fd, r);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions") == 0) {
        if (g_api_log) ylog_body("HTTP CHAT", req.body);
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_chat_completions(fd, r, req.body ? req.body : "", stream);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/completions") == 0) {
        if (g_api_log) ylog_body("HTTP COMPLETIONS", req.body);
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_completions(fd, r, req.body ? req.body : "", stream);
    } else {
        HttpResponse rr;
        http_begin(&rr, fd, 404, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"not found\"}}");
    }
    if (req.body) free(req.body);
    sock_close(fd);
}

/* HTTP 连接处理(每连接一线程: 长生成不阻塞其他连接) */
static Router* http_conn_router;
static void* http_conn(void* arg)
{
    int fd = (int)(intptr_t)arg;
    Router* r = http_conn_router;
    handle_conn(fd, r);
    return NULL;
}

static void http_thread(void* arg)
{
    HttpCtx* c = (HttpCtx*)arg;
    sock_init();
    int srv = sock_listen(c->port, 16);
    if (srv < 0) { ylog_error("http: cannot listen on %u", c->port); return; }
    ylog_info("http: OpenAI-compatible listening on port %u", c->port);
    http_conn_router = c->router;
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            pthread_t t;
            pthread_create(&t, NULL, http_conn, (void*)(intptr_t)fd);
            pthread_detach(t);
        }
    }
    sock_close(srv);
}

/* 启动 HTTP 线程(由 router --http-port 调用) */
int router_http_start(Router* r, uint16_t http_port)
{
    static HttpCtx ctx;
    ctx.router = r;
    ctx.port = http_port;
    void* t = NULL;
    return ythread_create(&t, http_thread, &ctx);
}
