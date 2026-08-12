#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    const char* key;
    const char* val;
} Arg;

static const char* opt(Arg* args, int n, const char* key, const char* def)
{
    int i;
    for (i = 0; i < n; i++) {
        if (strcmp(args[i].key, key) == 0) return args[i].val;
    }
    return def;
}

static int parse_args(int argc, char** argv, int start, Arg* args, int maxn)
{
    int n = 0;
    int i;
    for (i = start; i + 1 < argc; i += 2) {
        if (argv[i][0] != '-' || n >= maxn) break;
        args[n].key = argv[i];
        while (*args[n].key == '-') args[n].key++;
        args[n].val = argv[i + 1];
        n++;
    }
    return n;
}

static void on_token_cb(uint32_t id, void* ctx)
{
    Vocab* v = (Vocab*)ctx;
    char tmp[65536];
    vocab_decode(v, &id, 1, tmp, sizeof(tmp));
    fputs(tmp, stdout);
    fflush(stdout);
}

static int cmd_convert(int argc, char** argv)
{
    Arg a[16];
    int n = parse_args(argc, argv, 2, a, 16);
    const char* in = opt(a, n, "safetensors", NULL);
    const char* gguf = opt(a, n, "gguf", NULL);
    const char* out = opt(a, n, "out", NULL);
    const char* vocab_out = opt(a, n, "vocab", NULL);
    uint32_t seq = (uint32_t)atoi(opt(a, n, "seq", "2048"));
    uint32_t blocks = (uint32_t)atoi(opt(a, n, "blocks", "2"));
    uint32_t hidden = (uint32_t)atoi(opt(a, n, "hidden", "64"));
    uint32_t heads = (uint32_t)atoi(opt(a, n, "heads", "4"));
    uint32_t kv_heads = (uint32_t)atoi(opt(a, n, "kv-heads", "2"));
    uint32_t vocab = (uint32_t)atoi(opt(a, n, "vocab-size", "1024"));
    uint32_t seed = (uint32_t)atoi(opt(a, n, "seed", "42"));
    char err[1024];

    if (gguf && out) {
        if (convert_model("gguf", gguf, out, vocab_out, seq, err, sizeof(err)) != 0) {
            fprintf(stderr, "convert failed: %s\n", err);
            return 1;
        }
        printf("converted %s -> %s (max_seq %u)\n", gguf, out, seq);
        llf_check(out, err, sizeof(err));
        return 0;
    }
    if (in && out) {
        if (convert_model("safetensors", in, out, vocab_out, seq, err, sizeof(err)) != 0) {
            fprintf(stderr, "convert failed: %s\n", err);
            return 1;
        }
        printf("converted %s -> %s (dtype fp16, max_seq %u)\n", in, out, seq);
        llf_check(out, err, sizeof(err));
        return 0;
    }
    if (out) {
        if (convert_dummy(out, blocks, hidden, heads, kv_heads, vocab, seq, seed, err, sizeof(err)) != 0) {
            fprintf(stderr, "convert failed: %s\n", err);
            return 1;
        }
        printf("dummy model written: %s (blocks=%u hidden=%u heads=%u kv=%u vocab=%u seq=%u)\n",
               out, blocks, hidden, heads, kv_heads, vocab, seq);
        if (vocab_out) {
            if (dummy_vocab(vocab_out, vocab, err, sizeof(err)) != 0) {
                fprintf(stderr, "vocab failed: %s\n", err);
                return 1;
            }
            printf("vocab written: %s\n", vocab_out);
        }
        return 0;
    }
    if (vocab_out && !in) {
        if (dummy_vocab(vocab_out, vocab, err, sizeof(err)) != 0) {
            fprintf(stderr, "vocab failed: %s\n", err);
            return 1;
        }
        printf("vocab written: %s\n", vocab_out);
        return 0;
    }
    fprintf(stderr, "usage: yllm convert --safetensors <file> --out <file.llf> [--seq 2048]\n");
    fprintf(stderr, "   or: yllm convert --gguf <file.gguf> --out <file.llf> [--vocab <file.txt>] [--seq 2048]\n");
    fprintf(stderr, "   or: yllm convert --out <file.llf> [--blocks B --hidden H --heads Hh --kv-heads K --vocab-size V --seq S --seed N] [--vocab <file.txt>]\n");
    return 1;
}

static int cmd_check(int argc, char** argv)
{
    Arg a[8];
    int n = parse_args(argc, argv, 2, a, 8);
    const char* m = opt(a, n, "model", NULL);
    if (!m) {
        fprintf(stderr, "usage: yllm check --model <file.llf>\n");
        return 1;
    }
    char err[1024];
    if (llf_check(m, err, sizeof(err)) != 0) {
        fprintf(stderr, "check failed: %s\n", err);
        return 1;
    }
    return 0;
}

static int cmd_gen(int argc, char** argv)
{
    Arg a[16];
    int n = parse_args(argc, argv, 2, a, 16);
    const char* m = opt(a, n, "model", NULL);
    const char* vocab = opt(a, n, "vocab", "vocab.txt");
    const char* prompt = opt(a, n, "prompt", "Once upon a time");
    int ntokens = atoi(opt(a, n, "tokens", "64"));
    int budget_mb = atoi(opt(a, n, "budget-mb", "0"));
    int depth = atoi(opt(a, n, "depth", "2"));
    float temp = (float)atof(opt(a, n, "temp", "1.0"));
    float top_p = (float)atof(opt(a, n, "top-p", "0.9"));
    uint64_t seed = (uint64_t)strtoull(opt(a, n, "seed", "42"), NULL, 10);

    if (!m) {
        fprintf(stderr, "usage: yllm gen --model <file.llf> [--vocab <file>] [--prompt <text>] [--tokens N] [--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N]\n");
        return 1;
    }

    Vocab v;
    if (vocab_load(vocab, &v) != 0) {
        fprintf(stderr, "cannot load vocab: %s\n", vocab);
        return 1;
    }

    Engine e;
    char err[1024];
    uint64_t budget = (uint64_t)budget_mb * 1024 * 1024;
    if (engine_init(&e, m, budget, depth, err, sizeof(err)) != 0) {
        fprintf(stderr, "engine init failed: %s\n", err);
        vocab_free(&v);
        return 1;
    }

    uint32_t* ids = (uint32_t*)ymalloc((size_t)ntokens + 4096);
    uint32_t sz = (uint32_t)(ntokens + 4096);
    int nprompt = vocab_encode(&v, prompt, ids, (int)sz);
    printf("prompt tokens: %d\n", nprompt);
    printf("\n");
    uint64_t t0 = ynow_ms();
    int rc = 0;
    if (nprompt > 0) {
        rc = engine_generate(&e, ids, nprompt, ntokens, temp, top_p, seed, on_token_cb, &v, err, sizeof(err));
    }
    uint64_t ms = ynow_ms() - t0;
    printf("\n\n");
    printf("generated %d tokens in %.2f s\n", ntokens, (double)ms / 1000.0);
    if (ms > 0) printf("throughput: %.2f tok/s\n", (double)ntokens * 1000.0 / (double)ms);
    printf("resident estimate: %.2f MB (budget: %s)\n",
           (double)engine_resident(&e) / 1048576.0,
           budget_mb > 0 ? "limited" : "unlimited");
    if (rc != 0) fprintf(stderr, "generate failed: %s\n", err);
    engine_free(&e);
    vocab_free(&v);
    free(ids);
    return rc == 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: yllm <convert|check|gen> [options]\n");
        return 1;
    }
    if (strcmp(argv[1], "convert") == 0) return cmd_convert(argc, argv);
    if (strcmp(argv[1], "check") == 0) return cmd_check(argc, argv);
    if (strcmp(argv[1], "gen") == 0) return cmd_gen(argc, argv);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}