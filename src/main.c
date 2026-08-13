#include "yllm.h"
#include "dist.h"
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
    int rank = atoi(opt(a, n, "rank", "0"));
    int ranks = atoi(opt(a, n, "ranks", "1"));
    int port_base = atoi(opt(a, n, "port-base", "8900"));
    int dist_fp16 = atoi(opt(a, n, "dist-fp16", "0"));

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
    /* 分布式分片: 按字节均衡切层(末 rank 含 norm+head, 故少分块) */
    if (ranks > 1) {
        uint32_t blocks = e.ws.model.h.n_blocks;
        if ((uint32_t)ranks > blocks) { fprintf(stderr, "ranks %d > blocks %u\n", ranks, blocks); return 1; }
        uint32_t b;
        uint64_t total = 0;
        for (b = 1; b <= blocks; b++) total += e.ws.model.dir[b].size;
        uint64_t per = total / (uint32_t)ranks;
        uint32_t bs[64]; /* 每个 rank 的起始 block */
        bs[0] = 1;
        uint64_t acc = 0;
        int r = 1;
        for (b = 1; b <= blocks && r < ranks; b++) {
            acc += e.ws.model.dir[b].size;
            if (acc >= per * (uint64_t)r) { bs[r] = b + 1; r++; }
        }
        while (r < ranks) { bs[r] = blocks + 1; r++; }
        if (bs[ranks - 1] > blocks + 1) bs[ranks - 1] = blocks + 1;
        uint32_t begin = bs[rank];
        uint32_t end = rank + 1 < ranks ? bs[rank + 1] : e.ws.model.n_layers;
        if (rank == 0) begin = 0; /* rank0 含 embed */
        engine_set_layers(&e, begin, end);
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
        /* ================= 分布式层流水线 ================= */
        Dist dist;
        if (dist_init(&dist, rank, ranks, (uint16_t)port_base) != 0) {
            fprintf(stderr, "dist init failed\n");
            engine_free(&e);
            vocab_free(&v);
            free(ids);
            return 1;
        }
        uint32_t hidden = e.ws.model.h.hidden;
        uint32_t vocab_sz = e.ws.model.h.vocab;
        enum { TOPK = 1024 };
        float* xbuf = (float*)ymalloc((size_t)hidden * 4);
        uint32_t* k_ids = (uint32_t*)ymalloc((size_t)TOPK * 4);

        uint64_t rng = ysrand(seed);
        int ngen = 0;
        int dist_stats = getenv("YLLM_DIST_STATS") != NULL;
        const int STATS_EVERY = 8;

        if (rank == 0) {
            /* master: embed + 自己块段, 采样由收到的 top-k logits 决定 */
            uint32_t pos = 0;
            int i;
            for (i = 0; i < nprompt; i++) {
                engine_forward_range(&e, ids[i], 1, pos, xbuf, NULL);
                dist_send_x(&dist, pos, xbuf, hidden, dist_fp16);
                pos++;
            }
            float lse = 0.0f;
            /* 丢弃 prompt 阶段多余的 top-k(末 rank 对每个 X 帧都回 logits) */
            for (i = 0; i < nprompt - 1; i++) {
                if (dist_recv_logits(&dist, k_ids, e.logits, vocab_sz, &lse) <= 0) { rc = -1; break; }
            }
            for (i = 0; i < ntokens && rc == 0; i++) {
                if (pos >= e.max_seq) break;
                int k = dist_recv_logits(&dist, k_ids, e.logits, vocab_sz, &lse);
                if (k <= 0) { rc = -1; snprintf(err, sizeof(err), "dist recv logits failed"); break; }
                uint32_t nxt;
                if (engine_sample(&e, vocab_sz, temp, top_p, &rng, &nxt) != 0) { rc = -1; break; }
                if (getenv("YLLM_DISTDBG")) {
                    fprintf(stderr, "R0 sampled k=%d id=%u [%s] logit=%.2f\n", k, nxt,
                            v.pieces[nxt], e.logits[nxt]);
                }
                on_token_cb(nxt, &v);
                if (nxt == (uint32_t)v.eos) break;
                engine_forward_range(&e, nxt, 1, pos, xbuf, NULL);
                dist_send_x(&dist, pos, xbuf, hidden, dist_fp16);
                pos++;
                ngen++;
                if (dist_stats && (ngen % STATS_EVERY) == 0) {
                    char tag[48];
                    snprintf(tag, sizeof(tag), "dist@tok%d", ngen);
                    dist_print_stats(&dist, tag);
                }
            }
            dist_send_done(&dist);
            printf("\n\n");
            printf("decode:  %d tokens in %.2f s (%.1f tok/s)\n", ngen,
                   (double)(ynow_ms() - t0) / 1000.0,
                   (double)ngen * 1000.0 / (double)(ynow_ms() - t0 > 0 ? ynow_ms() - t0 : 1));
        } else {
            /* 中段/末段 rank: 收激活 → 算自己块段 → 转发/出 top-k logits */
            uint32_t pos;
            int t;
            int nf = 0;
            while ((t = dist_recv_x(&dist, &pos, xbuf, hidden, dist_fp16)) >= 0) {
                if (t == 3) { /* DONE: 向后转发并退出 */
                    dist_send_done(&dist);
                    break;
                }
                if (t != 1) break;
                nf++;
                memcpy(e.x, xbuf, (size_t)hidden * 4); /* 输入激活入引擎缓冲 */
                if (rank == ranks - 1) {
                    engine_forward_range(&e, 0, 0, pos, NULL, e.logits);
                    if (dist_send_logits(&dist, e.logits, vocab_sz, TOPK) != 0) { rc = -1; break; }
                } else {
                    engine_forward_range(&e, 0, 0, pos, xbuf, NULL);
                    if (dist_send_x(&dist, pos, xbuf, hidden, dist_fp16) != 0) { rc = -1; break; }
                }
                if (dist_stats && (nf % STATS_EVERY) == 0) {
                    char tag[48];
                    snprintf(tag, sizeof(tag), "dist@X%d", nf);
                    dist_print_stats(&dist, tag);
                }
            }
        }
        free(xbuf);
        free(k_ids);

        if (dist_stats || getenv("YLLM_DISTDBG")) dist_print_stats(&dist, "dist");
        dist_close(&dist);
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
    printf("\n\n");
    printf("prefill: %u tokens in %.2f s (%.1f tok/s)\n", tim.n_prefill,
           (double)tim.prefill_ms / 1000.0,
           tim.prefill_ms > 0 ? (double)tim.n_prefill * 1000.0 / (double)tim.prefill_ms : 0.0);
    printf("decode:  %u tokens in %.2f s (%.1f tok/s)\n", tim.n_decode,
           (double)tim.decode_ms / 1000.0,
           tim.decode_ms > 0 ? (double)tim.n_decode * 1000.0 / (double)tim.decode_ms : 0.0);
    printf("total:   %.2f s\n", (double)ms / 1000.0);
    printf("resident estimate: %.2f MB (budget: %s)\n",
           (double)engine_resident(&e) / 1048576.0,
           budget_mb > 0 ? "limited" : "unlimited");
    if (rc != 0) fprintf(stderr, "generate failed: %s\n", err);
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
    printf("chat prompt tokens: %d (bos=%d)\n", nprompt, v.bos);
    printf("\n");
    uint64_t t0 = ynow_ms();
    EngineTimings tim;
    memset(&tim, 0, sizeof(tim));
    int rc = 0;
    if (nprompt >= 0) {
        rc = engine_generate(&e, ids, nprompt, ntokens, temp, top_p, seed, v.eos, on_token_cb, &v, &tim, err, sizeof(err));
    }
    uint64_t ms = ynow_ms() - t0;
    printf("\n\n");
    printf("prefill: %u tokens in %.2f s (%.1f tok/s)\n", tim.n_prefill,
           (double)tim.prefill_ms / 1000.0,
           tim.prefill_ms > 0 ? (double)tim.n_prefill * 1000.0 / (double)tim.prefill_ms : 0.0);
    printf("decode:  %u tokens in %.2f s (%.1f tok/s)\n", tim.n_decode,
           (double)tim.decode_ms / 1000.0,
           tim.decode_ms > 0 ? (double)tim.n_decode * 1000.0 / (double)tim.decode_ms : 0.0);
    printf("total:   %.2f s\n", (double)ms / 1000.0);
    printf("resident estimate: %.2f MB (budget: %s)\n",
           (double)engine_resident(&e) / 1048576.0,
           budget_mb > 0 ? "limited" : "unlimited");
    if (rc != 0) fprintf(stderr, "chat failed: %s\n", err);
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
    if (strcmp(argv[1], "convert") == 0) return cmd_convert(argc, argv);
    if (strcmp(argv[1], "check") == 0) return cmd_check(argc, argv);
    if (strcmp(argv[1], "gen") == 0) return cmd_gen(argc, argv);
    if (strcmp(argv[1], "chat") == 0) return cmd_chat(argc, argv);
    fprintf(stderr, "unknown command: %s\n", argv[1]);
    return 1;
}