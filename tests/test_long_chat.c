/* 长序列 chat 吞吐: 同模型不同 prompt 长度测 prefill/decode。
 * 不进默认 make test(太慢, 需 gemma4 权重)。
 *
 *   make test-long-chat
 *   make test-long-chat LONG_N=14,512,2048,4096 LONG_DECODE=16
 *   ./build/avx2/test_long_chat.exe --model models/gemma-4-E2B-it-Q4_K_M.llf \
 *       --vocab models/gemma4.vocab.txt --n 14,512 --tokens 16
 */
#include "yllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* g_model = "models/gemma-4-E2B-it-Q4_K_M.llf";
static const char* g_vocab = "models/gemma4.vocab.txt";
static int g_tokens = 16;
static int g_use_chat = 1;

static const char* arg(int argc, char** argv, const char* key, const char* def)
{
    int i;
    size_t n = strlen(key);
    for (i = 1; i < argc; i++) {
        if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=' && argv[i][n + 1])
            return argv[i] + n + 1;
        if (strcmp(argv[i], key) == 0 && i + 1 < argc) return argv[++i];
    }
    return def;
}

static void make_text(char* dst, size_t cap)
{
    const char* fmt =
        "The history of village %d is a tale of farmers, rivers, forests, and quiet hills near the sea. ";
    size_t n = 0;
    int i = 0;
    dst[0] = 0;
    while (n + 160 < cap) {
        int w = snprintf(dst + n, cap - n, fmt, i++);
        if (w <= 0) break;
        n += (size_t)w;
    }
}

static int fill_prompt(Vocab* v, uint32_t* ids, int max, int want)
{
    char* text;
    int np, i;
    if (want <= 0 || want > max) return -1;
    text = (char*)malloc(65536);
    if (!text) return -1;
    make_text(text, 65536);
    if (g_use_chat)
        np = vocab_chat_ids(v, text, ids, max, 1);
    else
        np = vocab_encode(v, text, ids, max);
    free(text);
    if (np <= 0) return -1;
    if (np > want) return want;
    for (i = np; i < want; i++)
        ids[i] = ids[(i - 1) % np];
    return want;
}

static void run_one(Engine* e, Vocab* v, uint32_t* ids, int max_ids, int n_prompt)
{
    EngineTimings tim;
    char err[512];
    int np, rc;
    double pre_s, dec_s;
    np = fill_prompt(v, ids, max_ids, n_prompt);
    if (np < 0) {
        printf("FAIL: encode n=%d\n", n_prompt);
        return;
    }
    if ((uint32_t)np > e->max_seq) {
        printf("SKIP: n=%d > max_seq=%u\n", np, e->max_seq);
        return;
    }
    memset(&tim, 0, sizeof(tim));
    rc = engine_generate(e, ids, np, g_tokens, 0.0f, 1.0f, 42ull, -1,
                         NULL, NULL, &tim, err, sizeof(err));
    if (rc != 0) {
        printf("FAIL: generate n=%d (%s)\n", np, err);
        return;
    }
    pre_s = (double)tim.prefill_ms / 1000.0;
    dec_s = (double)tim.decode_ms / 1000.0;
    printf("n_prompt=%u  prefill: %u tok in %.2f s (%.2f tok/s)  decode: %u tok in %.2f s (%.2f tok/s)\n",
           tim.n_prefill, tim.n_prefill, pre_s,
           (pre_s > 0.0 && tim.n_prefill) ? (double)tim.n_prefill / pre_s : 0.0,
           tim.n_decode, dec_s,
           (dec_s > 0.0 && tim.n_decode) ? (double)tim.n_decode / dec_s : 0.0);
    fflush(stdout);
}

int main(int argc, char** argv)
{
    const char* ns;
    Engine e;
    Vocab v;
    char err[1024];
    uint32_t* ids;
    int max_ids, n;
    char* p;
    char buf[256];

    g_model = arg(argc, argv, "--model", g_model);
    g_vocab = arg(argc, argv, "--vocab", g_vocab);
    g_tokens = atoi(arg(argc, argv, "--tokens", "16"));
    g_use_chat = atoi(arg(argc, argv, "--chat", "1"));
    ns = arg(argc, argv, "--n", "14,512");

    if (vocab_load(g_vocab, &v) != 0) {
        printf("SKIP: cannot load vocab %s\n", g_vocab);
        return 0;
    }
    if (engine_init(&e, g_model, 0, 2, err, sizeof(err)) != 0) {
        printf("SKIP: engine init failed (%s)\n", err);
        vocab_free(&v);
        return 0;
    }
    max_ids = (int)e.max_seq;
    ids = (uint32_t*)malloc((size_t)max_ids * 4);
    if (!ids) {
        engine_free(&e);
        vocab_free(&v);
        return 1;
    }
    printf("model=%s max_seq=%u decode=%d chat_template=%d\n",
           g_model, e.max_seq, g_tokens, g_use_chat);
    snprintf(buf, sizeof(buf), "%s", ns);
    for (p = strtok(buf, ","); p; p = strtok(NULL, ",")) {
        n = atoi(p);
        if (n > 0) run_one(&e, &v, ids, max_ids, n);
    }
    free(ids);
    engine_free(&e);
    vocab_free(&v);
    return 0;
}
