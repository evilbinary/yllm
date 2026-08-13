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

#define HTTP_MAX_BODY (1 << 20)

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

/* SSE 回调: 逐 token 写 data 块 */
static void sse_on_token(const char* utf8, size_t len, void* ctx)
{
    HttpResponse* r = (HttpResponse*)ctx;
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
    char json[512];
    snprintf(json, sizeof(json),
             "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
             "\"created\":%lld,\"model\":\"yllm\","
             "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
             (long long)time(NULL), escaped);
    http_sse_data(r, json);
    free(escaped);
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
    int max_tokens = 32;
    if (json_find(body, "max_tokens", &v) && v.type == JSON_NUM)
        max_tokens = (int)json_num(&v);

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
        http_sse_begin(&rr, fd);
        int rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &rr);
        if (rc != 0)
            http_sse_data(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
        http_sse_done(&rr);
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        int rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc);
        if (rc != 0) {
            HttpResponse rr;
            http_begin(&rr, fd, 502, "application/json");
            http_reply(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
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
        free(json);
    }
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
    int max_tokens = 32;
    if (json_find(body, "max_tokens", &v) && v.type == JSON_NUM)
        max_tokens = (int)json_num(&v);

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
        http_sse_begin(&rr, fd);
        int rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &rr);
        if (rc != 0)
            http_sse_data(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
        http_sse_done(&rr);
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        int rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc);
        if (rc != 0) {
            HttpResponse rr;
            http_begin(&rr, fd, 502, "application/json");
            http_reply(&rr, "{\"error\":{\"message\":\"inference failed: backend timeout/disconnect\"}}");
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
        close(fd);
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
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_chat_completions(fd, r, req.body ? req.body : "", stream);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/completions") == 0) {
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_completions(fd, r, req.body ? req.body : "", stream);
    } else {
        HttpResponse rr;
        http_begin(&rr, fd, 404, "application/json");
        http_reply(&rr, "{\"error\":{\"message\":\"not found\"}}");
    }
    if (req.body) free(req.body);
    close(fd);
}

static void http_thread(void* arg)
{
    HttpCtx* c = (HttpCtx*)arg;
    sock_init();
    int srv = sock_listen(c->port, 16);
    if (srv < 0) { ylog_error("http: cannot listen on %u", c->port); return; }
    ylog_info("http: OpenAI-compatible listening on port %u", c->port);
    for (;;) {
        int fd = sock_accept_with_timeout(srv, 500);
        if (fd >= 0) {
            handle_conn(fd, c->router);
        }
    }
    close(srv);
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
