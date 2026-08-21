/* platform/android/gen_main.c — 精简 gen 入口(无 serve) */
#include "yllm.h"
#include "dist.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* key;
    const char* val;
} Arg;

static const char* opt(Arg* args, int n, const char* key, const char* def)
{
    int i;
    for (i = 0; i < n; i++)
        if (strcmp(args[i].key, key) == 0) return args[i].val;
    return def;
}

static int parse_args(int argc, char** argv, int start, Arg* args, int maxn)
{
    int n = 0, i;
    for (i = start; i + 1 < argc; i += 2) {
        if (argv[i][0] != '-' || n >= maxn) break;
        args[n].key = argv[i];
        while (*args[n].key == '-') args[n].key++;
        args[n].val = argv[i + 1];
        n++;
    }
    return n;
}

static int on_token_cb(uint32_t id, void* ctx)
{
    Vocab* v = (Vocab*)ctx;
    char tmp[65536];
    vocab_decode(v, &id, 1, tmp, sizeof(tmp));
    fputs(tmp, stdout);
    fflush(stdout);
    return 0;
}

static uint64_t budget_bytes_from_str(const char* model, const char* s)
{
    (void)model;
    if (!s || !s[0] || strcmp(s, "auto") == 0) return 0;
    char* end = NULL;
    double v = strtod(s, &end);
    if (end == s) return 0;
    while (*end == ' ') end++;
    if (*end == 'G' || *end == 'g') return (uint64_t)(v * 1024.0 * 1024.0 * 1024.0);
    if (*end == 'M' || *end == 'm') return (uint64_t)(v * 1024.0 * 1024.0);
    if (*end == 'K' || *end == 'k') return (uint64_t)(v * 1024.0);
    return (uint64_t)v;
}

int main(int argc, char** argv)
{
    Arg a[64];
    int n = parse_args(argc, argv, 1, a, 64);
    const char* m = opt(a, n, "model", NULL);
    const char* vocab = opt(a, n, "vocab", NULL);
    const char* prompt = opt(a, n, "prompt", "Hi");
    int ntokens = atoi(opt(a, n, "tokens", "16"));
    const char* budget_str = opt(a, n, "budget", "auto");
    int depth = atoi(opt(a, n, "depth", "2"));
    float temp = (float)atof(opt(a, n, "temp", "0"));
    float top_p = (float)atof(opt(a, n, "top-p", "0.9"));
    uint64_t seed = (uint64_t)strtoull(opt(a, n, "seed", "42"), NULL, 10);
    const char* device_s = opt(a, n, "device", "cpu");
    int gpu_id = atoi(opt(a, n, "gpu", "0"));

    if (!m || !vocab) {
        fprintf(stderr,
                "usage: yllm_gen --model <f.llf> --vocab <v> [--prompt t] [--tokens N] "
                "[--device cpu|vulkan] [--budget auto|NMB]\n");
        return 1;
    }

    ylog_open(NULL);
    Vocab v;
    if (vocab_load(vocab, &v) != 0) {
        fprintf(stderr, "vocab load failed: %s\n", vocab);
        return 1;
    }

    Engine e;
    char err[1024];
    uint64_t budget = budget_bytes_from_str(m, budget_str);
    if (engine_init(&e, m, budget, depth, err, sizeof(err)) != 0) {
        fprintf(stderr, "engine init: %s\n", err);
        vocab_free(&v);
        return 1;
    }

    DeviceKind dk = DEV_CPU;
    if (device_kind_parse(device_s, &dk) != 0) {
        fprintf(stderr, "bad --device %s\n", device_s);
        engine_free(&e);
        vocab_free(&v);
        return 1;
    }
    if (dk != DEV_CPU) {
        if (engine_bind_device(&e, dk, gpu_id, err, sizeof(err)) != 0) {
            fprintf(stderr, "bind device: %s\n", err);
            engine_free(&e);
            vocab_free(&v);
            return 1;
        }
    }

    uint32_t* ids = (uint32_t*)ymalloc(((size_t)ntokens + 4096) * 4);
    int nprompt = vocab_encode(&v, prompt, ids, ntokens + 4096);
    printf("prompt tokens: %d\n\n", nprompt);

    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    int eos = v.eos >= 0 ? v.eos : -1;
    if (engine_generate(&e, ids, nprompt, ntokens, temp, top_p, seed, eos,
                        on_token_cb, &v, &tim, err, sizeof(err)) != 0) {
        fprintf(stderr, "generate failed: %s\n", err);
    }
    printf("\n");
    ylog_info("prefill: %u tokens in %.2f s (%.2f tok/s)", tim.n_prefill,
              (double)tim.prefill_ms / 1000.0,
              tim.prefill_ms > 0 ? tim.n_prefill * 1000.0 / (double)tim.prefill_ms : 0.0);
    ylog_info("decode:  %u tokens in %.2f s (%.2f tok/s)", tim.n_decode,
              (double)tim.decode_ms / 1000.0,
              tim.decode_ms > 0 ? tim.n_decode * 1000.0 / (double)tim.decode_ms : 0.0);

    free(ids);
    engine_free(&e);
    vocab_free(&v);
    return 0;
}
