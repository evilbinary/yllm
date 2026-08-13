/* json.h — 最小 JSON 解析/查询(OpenAI HTTP 层用)
 *
 * 只支持 OpenAI 请求/响应所需的子集:
 *   对象 {"k":v, ...} / 数组 [v, ...] / 字符串 / 数字 / true/false/null
 * 提供: json_find(obj, "a.b", &val) 路径查询(返回值的起点指针+类型)
 */

#ifndef YLLM_SERVE_JSON_H
#define YLLM_SERVE_JSON_H

#include <stddef.h>
#include <string.h>
#include <stdlib.h>

typedef enum {
    JSON_NULL = 0,
    JSON_BOOL,
    JSON_NUM,
    JSON_STR,
    JSON_ARR,
    JSON_OBJ,
} JsonType;

typedef struct {
    JsonType type;
    const char* start;   /* 值起点(字符串为内容起点, 不含引号) */
    size_t len;          /* 字符串长度; 其他类型为原始跨度 */
} JsonVal;

/* 跳过空白 */
static inline const char* json_skip_ws(const char* p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* 解析一个值: 返回指向值结束后的指针 */
static inline const char* json_parse(const char* p, JsonVal* v)
{
    p = json_skip_ws(p);
    v->start = p;
    if (*p == '{') {
        v->type = JSON_OBJ;
        int depth = 0;
        while (*p) {
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') { depth--; if (depth == 0) { p++; break; } }
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') { if (*p == '\\') p++; p++; }
                if (*p == '"') p++;
                continue;   /* 字符串后不 p++, 避免吞掉紧随的 ']' '}' 分隔符 */
            }
            p++;
        }
        v->len = (size_t)(p - v->start);
    } else if (*p == '[') {
        v->type = JSON_ARR;
        int depth = 0;
        while (*p) {
            if (*p == '[' || *p == '{') depth++;
            else if (*p == ']' || *p == '}') { depth--; if (depth == 0) { p++; break; } }
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') { if (*p == '\\') p++; p++; }
                if (*p == '"') p++;
                continue;   /* 同上: 字符串后直接回到分支判断 */
            }
            p++;
        }
        v->len = (size_t)(p - v->start);
    } else if (*p == '"') {
        v->type = JSON_STR;
        p++;
        v->start = p;
        const char* q = p;
        while (*q && *q != '"') { if (*q == '\\') q++; q++; }
        v->len = (size_t)(q - p);
        if (*q == '"') q++;
        p = q;
    } else if (*p == 't' || *p == 'f') {
        v->type = JSON_BOOL;
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
               *p != '\t' && *p != '\n') p++;
        v->len = (size_t)(p - v->start);
    } else if (*p == 'n') {
        v->type = JSON_NULL;
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
               *p != '\t' && *p != '\n') p++;
        v->len = (size_t)(p - v->start);
    } else {
        v->type = JSON_NUM;
        while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' &&
               *p != '\t' && *p != '\n') p++;
        v->len = (size_t)(p - v->start);
    }
    return p;
}

/* 在对象内按 key 查找: 返回值的 JsonVal, 找到返回 1 */
static inline int json_find_in_obj(const char* obj, const char* key, JsonVal* out)
{
    /* obj 指向 '{' */
    const char* p = obj;
    size_t klen = strlen(key);
    p = json_skip_ws(p + 1);   /* 跳过 '{' */
    if (*p == '}') return 0;
    for (;;) {
        if (*p != '"') return 0;
        const char* ks = p + 1;
        const char* ke = ks;
        while (*ke && *ke != '"') { if (*ke == '\\') ke++; ke++; }
        size_t kl = (size_t)(ke - ks);
        p = ke;
        if (*p == '"') p++;
        p = json_skip_ws(p);
        if (*p != ':') return 0;
        p = json_skip_ws(p + 1);
        p = json_parse(p, out);
        if (kl == klen && memcmp(ks, key, klen) == 0) return 1;
        p = json_skip_ws(p);
        if (*p == ',') { p = json_skip_ws(p + 1); continue; }
        return 0;
    }
}

/* 路径查询: json_find(obj, "messages.0.content", &v)
 * 段: 名字(对象键) 或 数字(数组下标) */
static inline int json_find(const char* obj, const char* path, JsonVal* out)
{
    const char* cur = obj;
    char seg[128];
    while (*path) {
        const char* dot = strchr(path, '.');
        size_t len = dot ? (size_t)(dot - path) : strlen(path);
        if (len >= sizeof(seg)) return 0;
        memcpy(seg, path, len);
        seg[len] = '\0';
        path = dot ? dot + 1 : path + len;

        JsonVal v;
        if (seg[0] >= '0' && seg[0] <= '9') {
            /* 数组下标 */
            if (cur[0] != '[') return 0;
            int idx = atoi(seg);
            const char* p = json_skip_ws(cur + 1);
            int i;
            for (i = 0; i < idx; i++) {
                /* 数组越界保护: 遇到 ']' 提前失败 */
                p = json_skip_ws(p);
                if (*p == ']') return 0;
                if (*p != '"' && *p != '{' && *p != '[' && *p != 't' &&
                    *p != 'f' && *p != 'n' && !(*p >= '0' && *p <= '9') && *p != '-')
                    return 0;
                p = json_parse(p, &v);          /* p 已是值结束后的位置 */
                p = json_skip_ws(p);
                if (*p == ',') p = json_skip_ws(p + 1);
                else if (*p == ']') return 0;
            }
            p = json_skip_ws(p);
            if (*p == ']') return 0;
            json_parse(p, &v);
            cur = v.start;
            if (!*path) { *out = v; return 1; }
        } else {
            JsonVal v2;
            if (!json_find_in_obj(cur, seg, &v2)) return 0;
            cur = v2.start;
            if (!*path) { *out = v2; return 1; }
        }
    }
    return 0;
}

/* 字符串值转义还原(就地, 处理 \" \\ \n \t 等), 返回字符串长度 */
static inline size_t json_str_unescape(const char* s, size_t len, char* out)
{
    size_t i, o = 0;
    for (i = 0; i < len; i++) {
        if (s[i] == '\\' && i + 1 < len) {
            switch (s[i + 1]) {
                case 'n': out[o++] = '\n'; break;
                case 't': out[o++] = '\t'; break;
                case 'r': out[o++] = '\r'; break;
                case '"': out[o++] = '"'; break;
                case '\\': out[o++] = '\\'; break;
                default: out[o++] = s[i + 1]; break;
            }
            i++;
        } else {
            out[o++] = s[i];
        }
    }
    out[o] = '\0';
    return o;
}

/* 数字解析辅助 */
static inline double json_num(const JsonVal* v)
{
    char tmp[64];
    size_t len = v->len < sizeof(tmp) - 1 ? v->len : sizeof(tmp) - 1;
    memcpy(tmp, v->start, len);
    tmp[len] = '\0';
    return atof(tmp);
}

#endif /* YLLM_SERVE_JSON_H */
