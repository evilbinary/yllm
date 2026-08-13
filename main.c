#include "inference/yllm.h"
#include "inference/dist.h"
#include "inference/log.h"
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
    ylog_raw("%s", tmp);
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
            ylog_error("convert failed: %s", err);
            return 1;
        }
        printf("converted %s -> %s (max_seq %u)\n", gguf, out, seq);
        llf_check(out, err, sizeof(err));
        return 0;
    }
    if (in && out) {
        if (convert_model("safetensors", in, out, vocab_out, seq, err, sizeof(err)) != 0) {
            ylog_error("convert failed: %s", err);
            return 1;
        }
        printf("converted %s -> %s (dtype fp16, max_seq %u)\n", in, out, seq);
        llf_check(out, err, sizeof(err));
        return 0;
    }
    if (out) {
        if (convert_dummy(out, blocks, hidden, heads, kv_heads, vocab, seq, seed, err, sizeof(err)) != 0) {
            ylog_error("convert failed: %s", err);
            return 1;
        }
        printf("dummy model written: %s (blocks=%u hidden=%u heads=%u kv=%u vocab=%u seq=%u)\n",
               out, blocks, hidden, heads, kv_heads, vocab, seq);
        if (vocab_out) {
            if (dummy_vocab(vocab_out, vocab, err, sizeof(err)) != 0) {
                ylog_error("vocab failed: %s", err);
                return 1;
            }
            printf("vocab written: %s\n", vocab_out);
        }
        return 0;
    }
    if (vocab_out && !in) {
        if (dummy_vocab(vocab_out, vocab, err, sizeof(err)) != 0) {
            ylog_error("vocab failed: %s", err);
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
        ylog_error("check failed: %s", err);
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
    int rank = atoi(opt(a, n, "rank", "0"));
    int ranks = atoi(opt(a, n, "ranks", "1"));
    int port_base = atoi(opt(a, n, "port-base", "8900"));
    int dist_fp16 = atoi(opt(a, n, "dist-fp16", "0"));
    const char* dist_addrs = opt(a, n, "dist-addrs", NULL);

    if (!m) {
        fprintf(stderr, "usage: yllm gen --model <file.llf> [--vocab <file>] [--prompt <text>] [--tokens N] [--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N]\n");
        fprintf(stderr, "   or: yllm gen --model <file.llf> --ranks N --rank R [--port-base P]  (分布式层流水线, 所有 rank 相同命令)\n");
        return 1;
    }
    if (rank < 0 || rank >= ranks || ranks < 1) {
        fprintf(stderr, "bad rank/ranks: rank=%d ranks=%d\n", rank, ranks);
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
    /* 分布式分片: 按字节均衡切层 */
    if (ranks > 1) {
        if (dist_split_layers(&e, rank, ranks) != 0) {
            engine_free(&e);
            vocab_free(&v);
            return 1;
        }
    }

    uint32_t* ids = (uint32_t*)ymalloc(((size_t)ntokens + 4096) * 4);
    uint32_t sz = (uint32_t)(ntokens + 4096);
    int nprompt = vocab_encode(&v, prompt, ids, (int)sz);
    printf("prompt tokens: %d\n", nprompt);
    printf("\n");
    uint64_t t0 = ynow_ms();
    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    int rc = 0;

    if (ranks > 1) {
        /* 分布式层流水线各 rank 均执行 dist_gen */
        rc = dist_gen(&e, &v, ids, nprompt, ntokens, temp, top_p, seed,
                      rank, ranks, port_base, dist_addrs, dist_fp16, t0, on_token_cb, &v);
        engine_free(&e);
        vocab_free(&v);
        free(ids);
        return rc == 0 ? 0 : 1;
    }

    if (nprompt >= 0) {
        if (getenv("YLLM_DISTDBG")) {
            /* 手动逐 token 采样, 打印 rng 值(与 engine_generate 等价) */
            uint64_t rng2 = ysrand(seed);
            uint32_t pos = 0;
            int i2;
            for (i2 = 0; i2 < nprompt; i2++) { engine_forward(&e, ids[i2], pos); pos++; }
            for (i2 = 0; i2 < ntokens && rc == 0; i2++) {
                uint32_t nxt;
                if (engine_sample(&e, e.ws.model.h.vocab, temp, top_p, &rng2, &nxt) != 0) { rc = -1; break; }
                fprintf(stderr, "SINGLE pos%u id=%u [%s]\n", pos, nxt, v.pieces[nxt]);
                if (nxt == (uint32_t)v.eos) break;
                engine_forward(&e, nxt, pos);
                pos++;
            }
            goto done_gen;
        }
        rc = engine_generate(&e, ids, nprompt, ntokens, temp, top_p, seed, -1, on_token_cb, &v, &tim, err, sizeof(err));
    }
    uint64_t ms = ynow_ms() - t0;
    ylog_info("prefill: %u tokens in %.2f s (%.1f tok/s)", tim.n_prefill,
            (double)tim.prefill_ms / 1000.0,
            tim.prefill_ms > 0 ? (double)tim.n_prefill * 1000.0 / (double)tim.prefill_ms : 0.0);
    ylog_info("decode:  %u tokens in %.2f s (%.1f tok/s)", tim.n_decode,
            (double)tim.decode_ms / 1000.0,
            tim.decode_ms > 0 ? (double)tim.n_decode * 1000.0 / (double)tim.decode_ms : 0.0);
    ylog_info("total:   %.2f s", (double)ms / 1000.0);
    ylog_info("resident estimate: %.2f MB (budget: %s)",
            (double)engine_resident(&e) / 1048576.0,
            budget_mb > 0 ? "limited" : "unlimited");
    if (rc != 0) ylog_error("generate failed: %s", err);
done_gen:;
    engine_free(&e);
    vocab_free(&v);
    free(ids);
    return rc == 0 ? 0 : 1;
}

static int cmd_chat(int argc, char** argv)
{
    Arg a[16];
    int n = parse_args(argc, argv, 2, a, 16);
    const char* m = opt(a, n, "model", NULL);
    const char* vocab = opt(a, n, "vocab", "vocab.txt");
    const char* prompt = opt(a, n, "prompt", NULL);
    int ntokens = atoi(opt(a, n, "tokens", "128"));
    int budget_mb = atoi(opt(a, n, "budget-mb", "0"));
    int depth = atoi(opt(a, n, "depth", "2"));
    float temp = (float)atof(opt(a, n, "temp", "0.8"));
    float top_p = (float)atof(opt(a, n, "top-p", "0.9"));
    uint64_t seed = (uint64_t)strtoull(opt(a, n, "seed", "42"), NULL, 10);
    int no_template = atoi(opt(a, n, "no-template", "0"));
    int no_bos = atoi(opt(a, n, "no-bos", "0"));

    if (!m) {
        fprintf(stderr, "usage: yllm chat --model <file.llf> --prompt <text> [--vocab <file>] [--tokens N] [--budget-mb N] [--depth N] [--temp F] [--top-p F] [--seed N] [--no-template 1] [--no-bos 1]\n");
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

    uint32_t* ids = (uint32_t*)ymalloc(((size_t)ntokens + 8192) * 4);
    uint32_t sz = (uint32_t)(ntokens + 8192);
    int nprompt;
    int use_bos = no_bos ? 0 : v.add_bos;
    if (prompt && vocab_has_template(&v) && !no_template) {
        nprompt = vocab_chat_ids(&v, prompt, ids, (int)sz, use_bos);
        if (nprompt <= 0) {
            fprintf(stderr, "chat template render failed, falling back to plain encode\n");
            nprompt = vocab_encode(&v, prompt, ids, (int)sz);
            if (use_bos && v.bos >= 0 && nprompt < (int)sz) {
                memmove(ids + 1, ids, (size_t)nprompt * 4);
                ids[0] = (uint32_t)v.bos;
                nprompt++;
            }
        }
    } else {
        nprompt = vocab_encode(&v, prompt ? prompt : "Hello", ids, (int)sz);
        if (use_bos && v.bos >= 0 && nprompt < (int)sz) {
            memmove(ids + 1, ids, (size_t)nprompt * 4);
            ids[0] = (uint32_t)v.bos;
            nprompt++;
        }
    }
    ylog_info("chat prompt tokens: %d (bos=%d)", nprompt, v.bos);
    uint64_t t0 = ynow_ms();
    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    int rc = 0;
    if (nprompt >= 0) {
        rc = engine_generate(&e, ids, nprompt, ntokens, temp, top_p, seed, v.eos, on_token_cb, &v, &tim, err, sizeof(err));
    }
    uint64_t ms = ynow_ms() - t0;
    ylog_info("prefill: %u tokens in %.2f s (%.1f tok/s)", tim.n_prefill,
           (double)tim.prefill_ms / 1000.0,
           tim.prefill_ms > 0 ? (double)tim.n_prefill * 1000.0 / (double)tim.prefill_ms : 0.0);
    ylog_info("decode:  %u tokens in %.2f s (%.1f tok/s)", tim.n_decode,
           (double)tim.decode_ms / 1000.0,
           tim.decode_ms > 0 ? (double)tim.n_decode * 1000.0 / (double)tim.decode_ms : 0.0);
    ylog_info("total:   %.2f s", (double)ms / 1000.0);
    ylog_info("resident estimate: %.2f MB (budget: %s)",
           (double)engine_resident(&e) / 1048576.0,
           budget_mb > 0 ? "limited" : "unlimited");
    if (rc != 0) ylog_error("chat failed: %s", err);
    engine_free(&e);
    vocab_free(&v);
    free(ids);
    return rc == 0 ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: yllm <convert|check|gen|chat> [options]\n");
        return 1;
    }
    /* 全局日志: --log <file> [--log-level debug|info|warn|error] [--no-console] */
    const char* log_path = NULL;
    const char* log_level = NULL;
    int no_console = 0;
    int i;
    for (i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--log") == 0) { log_path = argv[i + 1]; i++; }
        else if (strcmp(argv[i], "--log-level") == 0) { log_level = argv[i + 1]; i++; }
        else if (strcmp(argv[i], "--no-console") == 0) no_console = 1;
    }
    ylog_open(log_path);
    if (no_console) ylog_set_console(0);
    if (log_level) {
        if (strcmp(log_level, "debug") == 0) ylog_set_level(YLOG_DEBUG);
        else if (strcmp(log_level, "warn") == 0) ylog_set_level(YLOG_WARN);
        else if (strcmp(log_level, "error") == 0) ylog_set_level(YLOG_ERROR);
        else ylog_set_level(YLOG_INFO);
    }
    ylog_info("yllm start: %s", log_path ? log_path : "(console only)");

    int rc;
    if (strcmp(argv[1], "convert") == 0) rc = cmd_convert(argc, argv);
    else if (strcmp(argv[1], "check") == 0) rc = cmd_check(argc, argv);
    else if (strcmp(argv[1], "gen") == 0) rc = cmd_gen(argc, argv);
    else if (strcmp(argv[1], "chat") == 0) rc = cmd_chat(argc, argv);
    else {
        fprintf(stderr, "unknown command: %s\n", argv[1]);
        rc = 1;
    }
    ylog_close();
    return rc;
}