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
#include "../inference/include/yllm.h"
#include "../inference/include/log.h"
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

/* OpenAI 兼容 API key(空 = 不校验) */
static char g_api_key[CFG_STR_MAX] = "";
void router_http_set_api_key(const char* key) {
    if (key) snprintf(g_api_key, sizeof(g_api_key), "%s", key);
    else g_api_key[0] = '\0';
}

/* 校验请求 key; 返回 0 通过, -1 拒绝 */
static int http_check_auth(const HttpRequest* req)
{
    if (!g_api_key[0]) return 0;   /* 未配置 key, 不校验 */
    char got[CFG_STR_MAX];
    if (http_auth_key(req, got, sizeof(got)) != 0) return -1;
    return strcmp(got, g_api_key) == 0 ? 0 : -1;
}

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

/* 把后端 ERR 正文转义进 JSON error.message(无正文时用 fallback) */
static void http_infer_error_json(char* out, size_t outsz, const char* msg, const char* fallback)
{
    const char* src = (msg && msg[0]) ? msg : fallback;
    size_t o = 0;
    const char* prefix = "{\"error\":{\"message\":\"";
    size_t pl = strlen(prefix);
    if (pl >= outsz) { if (outsz) out[0] = '\0'; return; }
    memcpy(out, prefix, pl);
    o = pl;
    for (; *src && o + 8 < outsz; src++) {
        char c = *src;
        if (c == '"' || c == '\\') {
            out[o++] = '\\';
            out[o++] = c;
        } else if (c == '\n') {
            out[o++] = '\\';
            out[o++] = 'n';
        } else if (c == '\r') {
            out[o++] = '\\';
            out[o++] = 'r';
        } else if ((unsigned char)c < 0x20) {
            /* 跳过其他控制字符 */
        } else {
            out[o++] = c;
        }
    }
    const char* suffix = "\"}}";
    size_t sl = strlen(suffix);
    if (o + sl >= outsz) o = outsz - sl - 1;
    memcpy(out + o, suffix, sl + 1);
}

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
    int prompt_tokens;   /* 会话模式真实 prompt token 数(server 渲染统计) */
    int include_usage;   /* 客户端请求 stream_options.include_usage */
    char pending[8];     /* 未完成 UTF-8 字符的尾字节缓冲(防跨 chunk 切分中文) */
    int pending_len;
} SseCtx;

/* 返回 buf[0..len) 中完整 UTF-8 字符的字节数(能安全作为 SSE chunk 独立解码的部分)。
 * 保留末尾可能的不完整多字节序列(留到下次拼接)。 */
static size_t utf8_complete_len(const char* buf, size_t len)
{
    size_t i = 0;
    while (i < len) {
        unsigned char c = (unsigned char)buf[i];
        size_t need;
        if (c < 0x80) need = 1;
        else if ((c >> 5) == 0x6) need = 2;
        else if ((c >> 4) == 0xE) need = 3;
        else if ((c >> 3) == 0x1E) need = 4;
        else { i++; continue; }   /* 非法字节, 当 1 字节 */
        if (i + need > len) break;   /* 末尾不完整, 留到下次 */
        i += need;
    }
    return i;
}

static void sse_on_token(const char* utf8, size_t len, void* ctx)
{
    SseCtx* sc = (SseCtx*)ctx;
    HttpResponse* r = sc->r;
    sc->n_tokens++;
    /* 拼上残留的不完整 UTF-8 序列 */
    char buf[512];
    size_t blen = 0;
    if (sc->pending_len) { memcpy(buf, sc->pending, sc->pending_len); blen = sc->pending_len; }
    if (blen + len < sizeof(buf)) { memcpy(buf + blen, utf8, len); blen += len; }
    else { blen = 0; }   /* 超缓冲, 丢弃(罕见) */
    size_t complete = utf8_complete_len(buf, blen);
    /* 保存不完整尾部 */
    sc->pending_len = (int)(blen - complete);
    if (sc->pending_len > 0) memcpy(sc->pending, buf + complete, sc->pending_len);
    if (complete == 0) return;   /* 全是不完整序列, 等下次 */

    char* escaped = (char*)malloc(complete * 2 + 1);
    if (!escaped) return;
    size_t o = 0, i;
    for (i = 0; i < complete; i++) {
        if (buf[i] == '"') { escaped[o++] = '\\'; escaped[o++] = '"'; }
        else if (buf[i] == '\\') { escaped[o++] = '\\'; escaped[o++] = '\\'; }
        else if (buf[i] == '\n') { escaped[o++] = '\\'; escaped[o++] = 'n'; }
        else if (buf[i] == '\r') { escaped[o++] = '\\'; escaped[o++] = 'r'; }
        else if (buf[i] == '\t') { escaped[o++] = '\\'; escaped[o++] = 't'; }
        else escaped[o++] = buf[i];
    }
    escaped[o] = '\0';
    size_t jlen = 256 + complete * 2;
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
        int pt = sc->prompt_tokens > 0 ? sc->prompt_tokens : 0;
        snprintf(json, sizeof(json),
                 "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                 (long long)time(NULL), sc->model, pt, ct, pt + ct);
        http_sse_data(sc->r, json);
    }
}

/* 会话 key = 客户端显式 id(头或 JSON); 没有则空, 由 server 按对话 token 前缀续接 */
static void pick_client_sess_id(const HttpRequest* req, const char* body, char* out, size_t outsz)
{
    JsonVal v;
    out[0] = '\0';
    if (req) {
        if (http_header_get(req, "X-Session-Id", out, outsz) == 0 && out[0]) return;
        if (http_header_get(req, "X-Conversation-Id", out, outsz) == 0 && out[0]) return;
    }
    if (body) {
        const char* keys[] = { "conversation_id", "session_id", "chat_id", NULL };
        int k;
        for (k = 0; keys[k]; k++) {
            if (json_find(body, keys[k], &v) && v.type == JSON_STR && v.len > 0) {
                size_t n = json_str_unescape(v.start, v.len, out);
                if (n >= outsz) n = outsz - 1;
                out[n] = '\0';
                if (out[0]) return;
            }
        }
    }
}

/* 打包全部 messages 给 server(长度前缀, 可含换行) */
static int pack_chat_msgs(char* out, size_t cap, char** roles, char** contents, int n, size_t* olen)
{
    size_t o = 0;
    int i, w;
    w = snprintf(out, cap, "MSGS %d\n", n);
    if (w < 0 || (size_t)w >= cap) return -1;
    o = (size_t)w;
    for (i = 0; i < n; i++) {
        const char* role = roles[i] ? roles[i] : "user";
        const char* c = contents[i] ? contents[i] : "";
        size_t cl = strlen(c);
        w = snprintf(out + o, cap - o, "%s %zu\n", role, cl);
        if (w < 0 || o + (size_t)w + cl > cap) return -1;
        o += (size_t)w;
        memcpy(out + o, c, cl);
        o += cl;
    }
    *olen = o;
    return 0;
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
static void handle_chat_completions(int fd, Router* r, const char* body, int stream, const HttpRequest* httpreq)
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
        http_begin(&rr, fd, 500, NULL);
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
    char* roles[32];
    char* contents[32];
    int n_msgs = 0;
    int sess_ok = 0;
    char sess_key[128] = "";
    char* packed = NULL;
    size_t packlen = 0;
    if (pool) {
        n_msgs = extract_chat_messages(body, pool, HTTP_MAX_BODY, roles, contents, 32);
        if (n_msgs > 0) {
            pick_client_sess_id(httpreq, body, sess_key, sizeof(sess_key));
            packed = (char*)malloc(HTTP_MAX_BODY);
            if (packed && pack_chat_msgs(packed, HTTP_MAX_BODY, roles, contents, n_msgs, &packlen) == 0)
                sess_ok = 1;
            else {
                free(packed);
                packed = NULL;
            }
        }
#if YLLM_SESS_DEBUG
        ylog_info("router_http: sess_ok=%d n_msgs=%d key=%s", sess_ok, n_msgs, sess_key[0] ? sess_key : "-");
#endif
    }

    if (plen == 0 && n_msgs == 0) {
        if (pool) free(pool);
        free(packed);
        free(prompt);
        free(collected);
        HttpResponse rr;
        http_begin(&rr, fd, 400, NULL);
        http_reply(&rr, "{\"error\":{\"message\":\"empty prompt\"}}");
        return;
    }

    if (stream) {
        HttpResponse rr;
        SseCtx sc;
        sc.r = &rr;
        sc.model = model;
        sc.n_tokens = 0;
        sc.prompt_tokens = 0;
        sc.include_usage = include_usage;
        if (g_api_log)
            ylog_info("HTTP CHAT stream model=%s max_tokens=%d temp=%.3g top_p=%.3g include_usage=%d sess_ok=%d n_msgs=%d plen=%zu key=%s",
                      model, max_tokens, rtemp, rtop_p, include_usage, sess_ok, n_msgs, plen,
                      sess_key[0] ? sess_key : "-");
        http_sse_begin(&rr, fd);
        int rc;
        char errbuf[256] = "";
        if (sess_ok)
            rc = router_infer_sess(r, model, max_tokens, sess_key,
                                   packed, packlen, sse_on_token, &sc,
                                   rtemp, rtop_p, &sc.prompt_tokens, errbuf, sizeof(errbuf));
        else
            rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &sc,
                              rtemp, rtop_p, errbuf, sizeof(errbuf));
        if (rc != 0) {
            if (rc == -2)
                http_sse_data(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            else {
                char ej[384];
                http_infer_error_json(ej, sizeof(ej), errbuf, "backend timeout/disconnect");
                http_sse_data(&rr, ej);
            }
        } else {
            sse_on_done(&sc, "stop");
        }
        http_sse_done(&rr);
        if (g_api_log) ylog_info("HTTP CHAT stream done rc=%d n_tokens=%d err=%s", rc, sc.n_tokens, errbuf[0] ? errbuf : "-");
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        int ptoken = 0;
        int rc;
        char errbuf[256] = "";
        if (sess_ok)
            rc = router_infer_sess(r, model, max_tokens, sess_key,
                                   packed, packlen, collect_on_token, &cc,
                                   rtemp, rtop_p, &ptoken, errbuf, sizeof(errbuf));
        else
            rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc,
                              rtemp, rtop_p, errbuf, sizeof(errbuf));
        if (rc != 0) {
            HttpResponse rr;
            if (rc == -2) {
                http_begin(&rr, fd, 404, NULL);
                http_reply(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            } else {
                char ej[384];
                http_infer_error_json(ej, sizeof(ej), errbuf, "backend timeout/disconnect");
                http_begin(&rr, fd, 502, NULL);
                http_reply(&rr, ej);
            }
            free(packed);
            if (pool) free(pool);
            free(prompt);
            free(collected);
            return;
        }
        char* json = (char*)malloc(HTTP_MAX_BODY + 512);
        if (!json) { free(packed); if (pool) free(pool); free(prompt); free(collected); return; }
        snprintf(json, HTTP_MAX_BODY + 512,
                 "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
                 "\"created\":%lld,\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},"
                 "\"finish_reason\":\"length\"}],"
                 "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                 (long long)time(NULL), model, collected, ptoken, cc.n_tokens, ptoken + cc.n_tokens);
        HttpResponse rr;
        http_begin(&rr, fd, 200, NULL);
        http_reply(&rr, json);
        if (g_api_log) { ylog_info("HTTP CHAT reply n_tokens=%d ptoken=%d", cc.n_tokens, ptoken); ylog_body("HTTP CHAT reply-json", json); }
        free(json);
    }
    if (pool) free(pool);
    free(packed);
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
        http_begin(&rr, fd, 500, NULL);
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
        http_begin(&rr, fd, 400, NULL);
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
        char errbuf[256] = "";
        int rc = router_infer(r, model, max_tokens, prompt, plen, sse_on_token, &sc,
                              rtemp, rtop_p, errbuf, sizeof(errbuf));
        if (rc != 0) {
            if (rc == -2)
                http_sse_data(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            else {
                char ej[384];
                http_infer_error_json(ej, sizeof(ej), errbuf, "backend timeout/disconnect");
                http_sse_data(&rr, ej);
            }
        } else {
            sse_on_done(&sc, "length");
        }
        http_sse_done(&rr);
        if (g_api_log) ylog_info("HTTP COMPLETIONS stream done rc=%d n_tokens=%d err=%s", rc, sc.n_tokens, errbuf[0] ? errbuf : "-");
    } else {
        CollectCtx cc;
        cc.buf = collected;
        cc.cap = HTTP_MAX_BODY;
        cc.len = 0;
        cc.n_tokens = 0;
        collected[0] = '\0';
        char errbuf[256] = "";
        int rc = router_infer(r, model, max_tokens, prompt, plen, collect_on_token, &cc,
                              rtemp, rtop_p, errbuf, sizeof(errbuf));
        if (rc != 0) {
            HttpResponse rr;
            if (rc == -2) {
                http_begin(&rr, fd, 404, NULL);
                http_reply(&rr, "{\"error\":{\"message\":\"model not found\"}}");
            } else {
                char ej[384];
                http_infer_error_json(ej, sizeof(ej), errbuf, "backend timeout/disconnect");
                http_begin(&rr, fd, 502, NULL);
                http_reply(&rr, ej);
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
        http_begin(&rr, fd, 200, NULL);
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
    http_begin(&rr, fd, 200, NULL);
    http_reply(&rr, json);
}

static void handle_conn(int fd, Router* r)
{
    HttpRequest req;
    if (http_parse_request(fd, &req) != 0) {
        sock_close(fd);
        return;
    }

    /* OpenAI 兼容 key 校验(health 除外); 失败返回 401 */
    if (!(strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) &&
        http_check_auth(&req) != 0) {
        HttpResponse rr;
        http_begin(&rr, fd, 401, NULL);
        http_reply(&rr, "{\"error\":{\"message\":\"invalid api key\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}");
        if (req.body) free(req.body);
        sock_close(fd);
        return;
    }

    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/health") == 0) {
        HttpResponse rr;
        http_begin(&rr, fd, 200, NULL);
        http_reply(&rr, "{\"status\":\"ok\"}");
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(fd, r);
    } else if (strcmp(req.method, "POST") == 0 &&
               strcmp(req.path, "/v1/chat/completions") == 0) {
        if (g_api_log) ylog_body("HTTP CHAT", req.body);
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_chat_completions(fd, r, req.body ? req.body : "", stream, &req);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/completions") == 0) {
        if (g_api_log) ylog_body("HTTP COMPLETIONS", req.body);
        int stream = req.body && strstr(req.body, "\"stream\":true");
        handle_completions(fd, r, req.body ? req.body : "", stream);
    } else {
        HttpResponse rr;
        http_begin(&rr, fd, 404, NULL);
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
