/* 会话缓存(cache.c)单元测试 */
#include "cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s (line %d)\n", msg, __LINE__); fails++; } \
} while (0)

int main(void)
{
    SessCache c;
    sess_init(&c, 4);

    /* 新建 + 追加 + 读取 */
    SessVal* v = sess_put(&c, "s1");
    CHECK(v != NULL, "put s1");
    static const uint32_t a[] = {1, 2, 3, 4, 5};
    CHECK(sess_commit(v, a, 5) == 0, "commit 5");
    CHECK(v->n == 5, "n==5");

    /* get 返回同一条目 */
    SessVal* g = sess_get(&c, "s1");
    CHECK(g == v, "get same entry");

    /* 前缀匹配 */
    static const uint32_t p1[] = {1, 2, 3, 9, 9};
    CHECK(sess_prefix(v, p1, 5) == 3, "prefix 3");
    static const uint32_t p2[] = {1, 2, 3, 4, 5, 6, 7};
    CHECK(sess_prefix(v, p2, 7) == 5, "prefix full");
    static const uint32_t p3[] = {0, 2, 3};
    CHECK(sess_prefix(v, p3, 3) == 0, "prefix 0");

    /* 截断 */
    sess_truncate(v, 3);
    CHECK(v->n == 3, "truncate to 3");
    sess_append(v, 6);
    CHECK(v->n == 4 && v->tokens[3] == 6, "append 6");

    /* 多会话 + 独立序列 */
    SessVal* v2 = sess_put(&c, "s2");
    static const uint32_t b[] = {10, 20};
    sess_commit(v2, b, 2);
    CHECK(v2->n == 2, "s2 n==2");
    CHECK(sess_get(&c, "s1")->n == 4, "s1 不受影响");

    /* 不存在的 key */
    CHECK(sess_get(&c, "nope") == NULL, "get missing");

    /* LRU 淘汰: 容量 4, 塞 5 个 → 必有一个被淘汰 */
    sess_put(&c, "s3");
    sess_put(&c, "s4");
    sess_get(&c, "s1");   /* 刷新 s1 的 last_use */
    SessVal* s5 = sess_put(&c, "s5");   /* 超容量 → 淘汰最久未用 */
    CHECK(s5 != NULL, "s5 put");
    CHECK(sess_get(&c, "s5") != NULL, "s5 exists");
    int alive = (sess_get(&c, "s1") != NULL) + (sess_get(&c, "s2") != NULL) +
                (sess_get(&c, "s3") != NULL) + (sess_get(&c, "s4") != NULL);
    CHECK(alive == 3, "exactly one evicted");

    /* ---- 磁盘落盘 ---- */
    /* token 列表落盘/载入(用新条目, 避免 LRU 淘汰后的悬垂指针) */
    SessVal* v5 = sess_put(&c, "s9");
    static const uint32_t d5[] = {1, 2, 3, 4, 5, 6};
    sess_commit(v5, d5, 6);
    sess_save(v5, "/tmp/opencode/sess_test.sess");
    SessVal v3;
    memset(&v3, 0, sizeof(v3));
    CHECK(sess_load(&v3, "/tmp/opencode/sess_test.sess") == 0, "sess_load ok");
    CHECK(v3.n == 6, "loaded n==6");
    CHECK(v3.tokens[5] == 6, "loaded tokens");
    CHECK(memcmp(v3.tokens, v5->tokens, 6 * 4) == 0, "loaded == saved");
    free(v3.tokens);
    /* 错误 magic 拒绝 */
    {
        FILE* bad = fopen("/tmp/opencode/sess_test.sess", "wb");
        fwrite("NOPEXXXX", 1, 8, bad);
        fclose(bad);
        SessVal v4; memset(&v4, 0, sizeof(v4));
        CHECK(sess_load(&v4, "/tmp/opencode/sess_test.sess") != 0, "bad magic rejected");
    }

    /* KV 落盘/载入(最小 Engine 结构) */
    {
        LlfHeader hh; memset(&hh, 0, sizeof(hh));
        hh.n_blocks = 2;
        Engine ee; memset(&ee, 0, sizeof(ee));
        ee.kv_dim = 4;
        ee.max_seq = 8;
        ee.ws.model.h = hh;
        ee.kv = (uint16_t*)calloc((2 * hh.n_blocks + 1) * ee.max_seq * ee.kv_dim, 2);
        uint32_t total = (2 * hh.n_blocks + 1) * ee.max_seq * ee.kv_dim;
        for (uint32_t i = 0; i < total; i++) ee.kv[i] = (uint16_t)(i * 7);
        CHECK(sess_kv_save(&ee, 5, "/tmp/opencode/sess_test.kv") == 0, "kv_save ok");
        memset(ee.kv, 0, (size_t)total * 2);
        uint32_t rp = 0;
        CHECK(sess_kv_load(&ee, "/tmp/opencode/sess_test.kv", &rp) == 0, "kv_load ok");
        CHECK(rp == 5, "kv loaded pos");
        int ok = 1;
        for (uint32_t l = 1; l <= hh.n_blocks; l++)   /* 落盘覆盖块 1..nb */
            for (uint32_t p = 0; p < 5; p++)
                for (uint32_t j = 0; j < ee.kv_dim; j++) {
                    uint16_t* k = ee.kv + l * ee.max_seq * ee.kv_dim;
                    uint16_t* v2 = ee.kv + (hh.n_blocks + l) * ee.max_seq * ee.kv_dim;
                    if (k[p * ee.kv_dim + j] != (uint16_t)((l * ee.max_seq + p) * ee.kv_dim + j) * 7) ok = 0;
                    if (v2[p * ee.kv_dim + j] != (uint16_t)(((hh.n_blocks + l) * ee.max_seq + p) * ee.kv_dim + j) * 7) ok = 0;
                }
        CHECK(ok, "kv data roundtrip");
        free(ee.kv);
    }

    sess_free(&c);
    if (fails == 0) {
        printf("cache tests: all passed\n");
        return 0;
    }
    printf("cache tests: %d FAILED\n", fails);
    return 1;
}
