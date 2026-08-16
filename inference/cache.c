#include "cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void sess_init(SessCache* c, int cap)
{
    memset(c, 0, sizeof(*c));
    c->cap = cap > 0 ? cap : 16;
    c->v = (SessVal*)ycalloc((size_t)c->cap, sizeof(SessVal));
}

void sess_free(SessCache* c)
{
    int i;
    for (i = 0; i < c->n; i++) {
        free(c->v[i].tokens);
    }
    free(c->v);
    memset(c, 0, sizeof(*c));
}

SessVal* sess_get(SessCache* c, const char* key)
{
    int i;
    for (i = 0; i < c->n; i++) {
        if (strcmp(c->v[i].key, key) == 0) {
            c->v[i].last_use = ynow_ms();
            return &c->v[i];
        }
    }
    return NULL;
}

static SessVal* sess_new(SessCache* c)
{
    SessVal* v;
    int i;
    if (c->n >= c->cap) {
        /* LRU 淘汰: 找最久未用 */
        int old = 0;
        for (i = 1; i < c->n; i++)
            if (c->v[i].last_use < c->v[old].last_use) old = i;
        free(c->v[old].tokens);
        memset(&c->v[old], 0, sizeof(SessVal));
        v = &c->v[old];
    } else {
        v = &c->v[c->n++];
    }
    return v;
}

SessVal* sess_put(SessCache* c, const char* key)
{
    SessVal* v = sess_get(c, key);
    if (v) return v;
    v = sess_new(c);
    snprintf(v->key, sizeof(v->key), "%s", key);
    v->n = 0;
    v->cap = 256;
    v->tokens = (uint32_t*)ymalloc((size_t)v->cap * 4);
    v->last_use = ynow_ms();
    return v;
}

uint32_t sess_prefix(const SessVal* v, const uint32_t* req, uint32_t n)
{
    uint32_t m = v->n < n ? v->n : n;
    uint32_t i;
    for (i = 0; i < m; i++)
        if (v->tokens[i] != req[i]) break;
    return i;
}

void sess_truncate(SessVal* v, uint32_t n)
{
    if (n < v->n) v->n = n;
    v->last_use = ynow_ms();
}

static int sess_grow(SessVal* v, uint32_t need)
{
    if (need <= v->cap) return 0;
    uint32_t nc = v->cap ? v->cap : 256;
    while (nc < need) nc *= 2;
    uint32_t* nt = (uint32_t*)realloc(v->tokens, (size_t)nc * 4);
    if (!nt) return -1;
    v->tokens = nt;
    v->cap = nc;
    return 0;
}

int sess_commit(SessVal* v, const uint32_t* tok, uint32_t n)
{
    if (sess_grow(v, v->n + n) != 0) return -1;
    memcpy(v->tokens + v->n, tok, (size_t)n * 4);
    v->n += n;
    v->last_use = ynow_ms();
    return 0;
}

int sess_append(SessVal* v, uint32_t tok)
{
    if (sess_grow(v, v->n + 1) != 0) return -1;
    v->tokens[v->n++] = tok;
    v->last_use = ynow_ms();
    return 0;
}

void sess_evict(SessCache* c)
{
    if (c->n == 0) return;
    int old = 0, i;
    for (i = 1; i < c->n; i++)
        if (c->v[i].last_use < c->v[old].last_use) old = i;
    free(c->v[old].tokens);
    memset(&c->v[old], 0, sizeof(SessVal));
    /* 尾部条目前移填补空洞 */
    if (old != c->n - 1) {
        c->v[old] = c->v[c->n - 1];
        memset(&c->v[c->n - 1], 0, sizeof(SessVal));
    }
    c->n--;
}

/* ---- 磁盘落盘 ---- */

#define SESS_FILE_MAGIC "YLLMSESS1"   /* 会话 token 文件 */
#define SESS_KV_MAGIC  "YLLMSKV01"   /* KV 缓存文件 */

static int write_all(FILE* f, const void* p, size_t n)
{
    return fwrite(p, 1, n, f) == n ? 0 : -1;
}

static int read_all(FILE* f, void* p, size_t n)
{
    return fread(p, 1, n, f) == n ? 0 : -1;
}

int sess_save(const SessVal* v, const char* path)
{
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int rc = 0;
    if (write_all(f, SESS_FILE_MAGIC, 8) != 0) rc = -1;
    else if (write_all(f, &v->n, 4) != 0) rc = -1;
    else if (v->n > 0 && write_all(f, v->tokens, (size_t)v->n * 4) != 0) rc = -1;
    fclose(f);
    return rc;
}

int sess_load(SessVal* v, const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t n = 0;
    int rc = 0;
    if (read_all(f, magic, 8) != 0 || memcmp(magic, SESS_FILE_MAGIC, 8) != 0) rc = -1;
    else if (read_all(f, &n, 4) != 0) rc = -1;
    else {
        if (sess_grow(v, n) != 0) rc = -1;
        else if (n > 0 && read_all(f, v->tokens, (size_t)n * 4) != 0) rc = -1;
        else {
            v->n = n;
            v->last_use = ynow_ms();
        }
    }
    fclose(f);
    return rc;
}

int sess_kv_save(Engine* e, uint32_t pos, const char* path)
{
    const LlfHeader* h = &e->ws.model.h;
    uint32_t kv_dim = e->kv_dim, max_seq = e->max_seq, nb = h->n_blocks;
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    int rc = 0;
    if (write_all(f, SESS_KV_MAGIC, 8) != 0 ||
        write_all(f, &nb, 4) != 0 || write_all(f, &kv_dim, 4) != 0 ||
        write_all(f, &max_seq, 4) != 0 || write_all(f, &pos, 4) != 0)
        rc = -1;
    size_t row = (size_t)pos * kv_dim;
    uint32_t l;
    /* 前向用块 1..nb 的 kv 分区(区域 1..nb 为 K, nb+1..2nb 为 V); 区域 0 未用 */
    for (l = 1; !rc && l <= nb; l++) {
        const uint16_t* k = e->kv + (size_t)l * max_seq * kv_dim;
        const uint16_t* v = e->kv + (size_t)(h->n_blocks + l) * max_seq * kv_dim;
        if (write_all(f, k, row * 2) != 0 || write_all(f, v, row * 2) != 0) rc = -1;
    }
    fclose(f);
    return rc;
}

int sess_kv_load(Engine* e, const char* path, uint32_t* pos)
{
    const LlfHeader* h = &e->ws.model.h;
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t nb = 0, kv_dim = 0, max_seq = 0, pos0 = 0;
    int rc = 0;
    if (read_all(f, magic, 8) != 0 || memcmp(magic, SESS_KV_MAGIC, 8) != 0) rc = -1;
    else if (read_all(f, &nb, 4) != 0 || read_all(f, &kv_dim, 4) != 0 ||
             read_all(f, &max_seq, 4) != 0 || read_all(f, &pos0, 4) != 0)
        rc = -1;
    /* 结构与引擎不符 → 拒绝载入(防错模型/错配置) */
    else if (nb != h->n_blocks || kv_dim != e->kv_dim || max_seq != e->max_seq ||
             pos0 > max_seq)
        rc = -1;
    else {
        size_t row = (size_t)pos0 * kv_dim;
        uint32_t l;
        for (l = 1; !rc && l <= nb; l++) {
            uint16_t* k = e->kv + (size_t)l * max_seq * kv_dim;
            uint16_t* v = e->kv + (size_t)(h->n_blocks + l) * max_seq * kv_dim;
            if (read_all(f, k, row * 2) != 0 || read_all(f, v, row * 2) != 0) rc = -1;
        }
        if (!rc && pos) *pos = pos0;
    }
    fclose(f);
    return rc;
}
