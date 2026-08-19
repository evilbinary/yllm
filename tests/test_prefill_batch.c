#include "yllm.h"
#include <stdio.h>
#include <string.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)

/* ---- 批量 prefill 回归: 顺序 vs 批量 logits 必须一致 ----
 * 回归覆盖:
 *  - matmul_batch 输入布局: FFN down 的 in=inter≠hidden 曾按错误步长读输入
 *    (swiglu 输出必须是 [B×inter] 布局, 不是 [B×hidden]), 导致逐层放大成垃圾
 *  - 批量嵌入 / QKV / 注意力 / KV 缓存与单 token 路径的一致性
 *  - 超过批容量(64)时的分批处理(长 prompt)
 * 注意: 两个引擎必须独立初始化(结构体拷贝会共享指针, 无法对比)。
 */
static void test_prefill_batch(const char* model_path, const char* vocab_path)
{
#ifndef __AVX2__
    /* 批量路径的快速内核在 AVX2 构建里; 基础构建走通用逐元素路径太慢 */
    printf("SKIP: prefill batch test requires AVX2 build\n");
    return;
#else
    char err[1024];
    Vocab v;
    if (vocab_load(vocab_path, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", vocab_path);
        return;
    }
    static const char* prompts[] = {
        "Once upon a time",
        /* 长 prompt: 超过批量容量 64, 触发分批(15 段 ≈68 token, 2 批) */
        "Once upon a time Once upon a time Once upon a time Once upon a time Once upon a time "
        "Once upon a time Once upon a time Once upon a time Once upon a time Once upon a time "
        "Once upon a time Once upon a time Once upon a time Once upon a time Once upon a time",
    };
    int pi;
    for (pi = 0; pi < 2; pi++) {
        uint32_t ids[1024];
        int n = vocab_chat_ids(&v, prompts[pi], ids, 1024, v.add_bos);
        if (n <= 0) { g_fail++; printf("FAIL: chat ids empty (%s:%d)\n", __FILE__, __LINE__); continue; }

        Engine e1, e2;
        if (engine_init(&e1, model_path, 0, 2, err, sizeof(err)) != 0) {
            printf("SKIP: cannot load %s (%s)\n", model_path, err);
            vocab_free(&v);
            return;
        }
        if (engine_init(&e2, model_path, 0, 2, err, sizeof(err)) != 0) {
            printf("SKIP: cannot load %s (%s)\n", model_path, err);
            engine_free(&e1);
            vocab_free(&v);
            return;
        }

        /* 顺序 prefill */
        uint32_t pos = 0;
        uint32_t i;
        for (i = 0; i < (uint32_t)n; i++) { engine_forward(&e1, ids[i], pos); pos++; }
        /* 批量 prefill */
        engine_forward_prefill(&e2, ids, n, 0);

        /* logits 最大差
         * 容差 2.0: 顺序路径(matmul_q4k, int8 量化 x)与批量路径(matmul_batch,
         * float 点积)是两种合法算法, 固有差异经多层放大约 0.5~0.8 logit;
         * 本检查用于抓布局类 bug(差异达数百), 更紧的容差会产生假阴性。 */
        float maxdiff = 0.0f;
        for (i = 0; i < e1.ws.model.h.vocab; i++) {
            float d = e1.logits[i] - e2.logits[i];
            if (d < 0) d = -d;
            if (d > maxdiff) maxdiff = d;
        }
        {
            char msg[96];
            snprintf(msg, sizeof(msg), "prefill(%d tok): batch logits match sequential", n);
            CHECK(maxdiff < 2.0f, msg);
        }

        /* 贪心解码 5 步, token 序列必须一致 */
        {
            uint32_t p1 = (uint32_t)n, p2 = (uint32_t)n;
            uint64_t rng1 = 42, rng2 = 42;
            int match = 1;
            int step;
            for (step = 0; step < 5; step++) {
                uint32_t t1, t2;
                engine_sample(&e1, e1.ws.model.h.vocab, 0.0f, 0.9f, &rng1, &t1);
                engine_sample(&e2, e2.ws.model.h.vocab, 0.0f, 0.9f, &rng2, &t2);
                if (t1 != t2) { match = 0; break; }
                engine_forward(&e1, t1, p1); p1++;
                engine_forward(&e2, t2, p2); p2++;
            }
            {
                char msg[96];
                snprintf(msg, sizeof(msg), "prefill(%d tok): greedy decode tokens match", n);
                CHECK(match != 0, msg);
            }
        }

        engine_free(&e1);
        engine_free(&e2);
    }
    vocab_free(&v);
#endif
}

int main(int argc, char** argv)
{
    const char* model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf";
    const char* vocab_path = "models/tinyllama.vocab.txt";
    if (argc > 1) model_path = argv[1];
    if (argc > 2) vocab_path = argv[2];

    test_prefill_batch(model_path, vocab_path);
    printf("prefill batch tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
