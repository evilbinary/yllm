#include "yllm.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    const uint8_t* p;
    const uint8_t* end;
} PB;

static uint64_t pb_varint(PB* b)
{
    uint64_t v = 0;
    int sh = 0;
    while (b->p < b->end) {
        uint8_t c = *b->p++;
        v |= (uint64_t)(c & 0x7f) << sh;
        if (!(c & 0x80)) return v;
        sh += 7;
    }
    return v;
}

static int parse_sp(const uint8_t* data, uint64_t size, Vocab* v)
{
    uint32_t cap = 1024;
    v->pieces = (char**)ymalloc((size_t)cap * sizeof(char*));
    v->n = 0;
    v->unk = -1;
    v->bos = -1;
    v->eos = -1;
    PB b;
    b.p = data;
    b.end = data + size;
    while (b.p < b.end) {
        uint32_t tag = (uint32_t)pb_varint(&b);
        uint32_t f = tag >> 3;
        uint32_t wt = tag & 7;
        if (f == 1 && wt == 2) {
            uint64_t len = pb_varint(&b);
            if (b.p + len > b.end) break;
            PB sub;
            sub.p = b.p;
            sub.end = b.p + len;
            b.p += len;
            char* pc = NULL;
            uint32_t pct = 0;
            while (sub.p < sub.end) {
                uint32_t t2 = (uint32_t)pb_varint(&sub);
                uint32_t f2 = t2 >> 3;
                uint32_t w2 = t2 & 7;
                if (f2 == 1 && w2 == 2) {
                    uint64_t l2 = pb_varint(&sub);
                    if (sub.p + l2 > sub.end) break;
                    pc = (char*)ymalloc((size_t)l2 + 1);
                    memcpy(pc, sub.p, (size_t)l2);
                    pc[l2] = 0;
                    sub.p += l2;
                } else if (f2 == 3 && (t2 & 7) == 0) {
                    pct = (uint32_t)pb_varint(&sub);
                } else if (f2 == 2 && (t2 & 7) == 5) {
                    sub.p += 4;
                } else if ((t2 & 7) == 0) {
                    pb_varint(&sub);
                } else if ((t2 & 7) == 2) {
                    uint64_t l2 = pb_varint(&sub);
                    if (sub.p + l2 > sub.end) break;
                    sub.p += l2;
                } else if ((t2 & 7) == 5) {
                    sub.p += 4;
                } else if ((t2 & 7) == 1) {
                    sub.p += 8;
                }
            }
            if (!pc) continue;
            if (v->n == (int)cap) {
                cap *= 2;
                v->pieces = (char**)realloc(v->pieces, (size_t)cap * sizeof(char*));
            }
            v->pieces[v->n] = pc;
            if (pct == 2 && v->unk < 0) v->unk = v->n;
            if (pct == 3) {
                if (strcmp(pc, "<bos>") == 0) v->bos = v->n;
                else if (v->bos < 0 && strcmp(pc, "<s>") == 0) v->bos = v->n;
                if (strcmp(pc, "<eos>") == 0) v->eos = v->n;
                else if (v->eos < 0 && strcmp(pc, "</s>") == 0) v->eos = v->n;
                if (strcmp(pc, "<unk>") == 0 && v->unk < 0) v->unk = v->n;
            }
            v->n++;
        } else if ((tag & 7) == 2) {
            uint64_t l2 = pb_varint(&b);
            if (b.p + l2 > b.end) break;
            b.p += l2;
        } else if ((tag & 7) == 0) {
            pb_varint(&b);
        } else if ((tag & 7) == 5) {
            b.p += 4;
        } else if ((tag & 7) == 1) {
            b.p += 8;
        }
    }
    if (v->n == 0) return -1;
    return 0;
}

static int vocab_bsearch_sorted(const Vocab* v, const int* sorted, const char* str, size_t len, int* id)
{
    int lo = 0, hi = v->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        const char* pc = v->pieces[sorted[mid]];
        int c = strncmp(pc, str, len);
        if (c == 0) {
            if (pc[len] == 0) { *id = sorted[mid]; return 0; }
            /* pc is a proper prefix of str (pc shorter)? or str is a prefix of pc */
            /* strncmp compared only len bytes; pc[len]!=0 means pc is longer,
               so pc sorts after str -> search to the right */
            c = 1;
        }
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return -1;
}

static int order_compare(const void* a, const void* b);
static int dict_compare(const void* a, const void* b);

static char* unescape_piece(const char* s)
{
    if (!strchr(s, '\\')) return ystrdup(s);
    size_t n = strlen(s);
    char* out = (char*)ymalloc(n + 1);
    size_t o = 0;
    size_t i;
    for (i = 0; i < n; i++) {
        if (s[i] == '\\' && i + 1 < n) {
            if (s[i + 1] == 'n') { out[o++] = '\n'; i++; continue; }
            if (s[i + 1] == 'r') { out[o++] = '\r'; i++; continue; }
            if (s[i + 1] == 't') { out[o++] = '\t'; i++; continue; }
            if (s[i + 1] == '\\') { out[o++] = '\\'; i++; continue; }
        }
        out[o++] = s[i];
    }
    out[o] = 0;
    return out;
}

static uint32_t* g_ml;
static uint32_t* g_mr;
static int merge_cmp(const void* a, const void* b)
{
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    if (g_ml[x] != g_ml[y]) return g_ml[x] < g_ml[y] ? -1 : 1;
    if (g_mr[x] != g_mr[y]) return g_mr[x] < g_mr[y] ? -1 : 1;
    return 0;
}

static void build_byte_ids(Vocab* v);

static int parse_text(const char* path, Vocab* v)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char line[65536];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    int n = atoi(line);
    if (n <= 0 || n > 1000000) { fclose(f); return -1; }
    memset(v, 0, sizeof(*v));   /* 全字段清零: 无 #SCORES#/#MERGES# 段时不得读未初始化内存 */
    v->pieces = (char**)ycalloc((size_t)n, sizeof(char*));
    v->n = 0;
    v->unk = -1;
    v->bos = -1;
    v->eos = -1;
    while (v->n < n && fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        char* tab = strchr(line, '\t');
        if (!tab) { fclose(f); return -1; }
        *tab = 0;
        int id = atoi(line);
        const char* pc = tab + 1;
        v->pieces[v->n] = unescape_piece(pc);
        if (id == v->n) {
            if (strcmp(pc, "<unk>") == 0) v->unk = v->n;
            if (strcmp(pc, "<bos>") == 0) v->bos = v->n;
            else if (v->bos < 0 && strcmp(pc, "<s>") == 0) v->bos = v->n;
            if (strcmp(pc, "<eos>") == 0) v->eos = v->n;
            else if (v->eos < 0 && strcmp(pc, "</s>") == 0) v->eos = v->n;
        }
        v->n++;
    }
    if (v->n == 0) { fclose(f); return -1; }

    /* optional sections: #SCORES#, #MERGES#, #CHAT# (任意顺序/缺失) */
    for (;;) {
        if (!fgets(line, sizeof(line), f)) break;
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (strcmp(line, "#SCORES#") == 0) {
            if (!fgets(line, sizeof(line), f)) break;
            int ns = atoi(line);
            if (ns > 0 && ns <= 1000000) {
                v->scores = (float*)ymalloc((size_t)ns * sizeof(float));
                v->n_scores = (uint32_t)ns;
                uint32_t si = 0;
                while (si < v->n_scores && fgets(line, sizeof(line), f)) {
                    v->scores[si++] = (float)atof(line);
                }
            }
            continue;
        }
        if (strcmp(line, "#MERGES#") == 0) {
            if (!fgets(line, sizeof(line), f)) break;
            int nm = atoi(line);
            if (nm > 0 && nm <= 5000000) {
                v->mls = (char**)ycalloc((size_t)nm, sizeof(char*));
                v->mrs = (char**)ycalloc((size_t)nm, sizeof(char*));
                uint32_t mi = 0;
                char lr[65536];
                while (mi < (uint32_t)nm && fgets(lr, sizeof(lr), f)) {
                    size_t l2 = strlen(lr);
                    while (l2 > 0 && (lr[l2 - 1] == '\n' || lr[l2 - 1] == '\r')) lr[--l2] = 0;
                    char* sp = strchr(lr, ' ');
                    if (!sp) continue;
                    *sp = 0;
                    v->mls[mi] = unescape_piece(lr);
                    v->mrs[mi] = unescape_piece(sp + 1);
                    mi++;
                }
                v->n_merges = mi;
            }
            continue;
        }
        if (strcmp(line, "#CHAT#") == 0) {
            while (fgets(line, sizeof(line), f)) {
                size_t l2 = strlen(line);
                while (l2 > 0 && (line[l2 - 1] == '\n' || line[l2 - 1] == '\r')) line[--l2] = 0;
                if (strncmp(line, "add_bos=", 8) == 0) {
                    v->add_bos = atoi(line + 8);
                } else if (strncmp(line, "eos_id=", 7) == 0) {
                    v->eos = atoi(line + 7);
                } else if (strncmp(line, "bos_id=", 7) == 0) {
                    int id = atoi(line + 7);
                    if (id >= 0) v->bos = id;
                } else if (strncmp(line, "template=", 9) == 0) {
                    v->chat_template = unescape_piece(line + 9);
                }
            }
            break;
        }
        break;
    }
    fclose(f);
    return 0;
}

static int order_compare(const void* a, const void* b)
{
    const char* x = *(char* const*)a;
    const char* y = *(char* const*)b;
    size_t lx = strlen(x);
    size_t ly = strlen(y);
    if (lx != ly) return lx > ly ? -1 : 1;
    return strcmp(x, y);
}

static int dict_compare(const void* a, const void* b)
{
    const char* x = *(char* const*)a;
    const char* y = *(char* const*)b;
    return strcmp(x, y);
}

static char** g_pieces;
static int idx_order_cmp(const void* a, const void* b)
{
    return order_compare(&g_pieces[*(const int*)a], &g_pieces[*(const int*)b]);
}
static int idx_dict_cmp(const void* a, const void* b)
{
    return dict_compare(&g_pieces[*(const int*)a], &g_pieces[*(const int*)b]);
}

int vocab_load(const char* path, Vocab* v)
{
    memset(v, 0, sizeof(*v));
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    int first = fgetc(f);
    fclose(f);
    if (first == 0x0a) {
        uint64_t sz;
        if (yfile_size(path, &sz) != 0) return -1;
        uint8_t* data = (uint8_t*)ymalloc((size_t)sz);
        FILE* f2 = fopen(path, "rb");
        if (!f2) { free(data); return -1; }
        if (fread(data, 1, (size_t)sz, f2) != sz) { fclose(f2); free(data); return -1; }
        fclose(f2);
        int rc = parse_sp(data, sz, v);
        free(data);
        if (rc != 0) return -1;
    } else {
        if (parse_text(path, v) != 0) return -1;
    }
    v->order = (int*)ymalloc((size_t)v->n * sizeof(int));
    v->sorted = (int*)ymalloc((size_t)v->n * sizeof(int));
    int i;
    for (i = 0; i < v->n; i++) { v->order[i] = i; v->sorted[i] = i; }
    g_pieces = v->pieces;
    qsort(v->order, (size_t)v->n, sizeof(int), idx_order_cmp);
    qsort(v->sorted, (size_t)v->n, sizeof(int), idx_dict_cmp);
    /* 解析 merges 原始串 -> 词条 id(用 sorted 二分, 避免 O(n^2)) */
    if (v->n_merges > 0) {
        v->ml = (uint32_t*)ymalloc((size_t)v->n_merges * sizeof(uint32_t));
        v->mr = (uint32_t*)ymalloc((size_t)v->n_merges * sizeof(uint32_t));
        v->mid = (uint32_t*)ymalloc((size_t)v->n_merges * sizeof(uint32_t));
        v->mrank = (uint32_t*)ymalloc((size_t)v->n_merges * sizeof(uint32_t));
        uint32_t mi = 0;
        for (i = 0; i < (int)v->n_merges; i++) {
            int idl = -1, idr = -1, idm = -1;
            if (vocab_bsearch_sorted(v, v->sorted, v->mls[i], strlen(v->mls[i]), &idl) != 0) continue;
            if (vocab_bsearch_sorted(v, v->sorted, v->mrs[i], strlen(v->mrs[i]), &idr) != 0) continue;
            size_t ll = strlen(v->mls[i]), lr2 = strlen(v->mrs[i]);
            char* merged = (char*)ymalloc(ll + lr2 + 1);
            memcpy(merged, v->mls[i], ll);
            memcpy(merged + ll, v->mrs[i], lr2);
            merged[ll + lr2] = 0;
            int ok = vocab_bsearch_sorted(v, v->sorted, merged, ll + lr2, &idm);
            free(merged);
            if (ok != 0) continue;
            v->ml[mi] = (uint32_t)idl;
            v->mr[mi] = (uint32_t)idr;
            v->mid[mi] = (uint32_t)idm;
            v->mrank[mi] = mi;
            mi++;
        }
        v->n_merges = mi;
        /* sort by (ml, mr) */
        g_ml = v->ml;
        g_mr = v->mr;
        uint32_t* idx = (uint32_t*)ymalloc((size_t)mi * sizeof(uint32_t));
        uint32_t qi;
        for (qi = 0; qi < mi; qi++) idx[qi] = qi;
        qsort(idx, (size_t)mi, sizeof(uint32_t), merge_cmp);
        uint32_t* nml = (uint32_t*)ymalloc((size_t)mi * sizeof(uint32_t));
        uint32_t* nmr = (uint32_t*)ymalloc((size_t)mi * sizeof(uint32_t));
        uint32_t* nmid = (uint32_t*)ymalloc((size_t)mi * sizeof(uint32_t));
        uint32_t* nmrk = (uint32_t*)ymalloc((size_t)mi * sizeof(uint32_t));
        for (qi = 0; qi < mi; qi++) {
            nml[qi] = v->ml[idx[qi]];
            nmr[qi] = v->mr[idx[qi]];
            nmid[qi] = v->mid[idx[qi]];
            nmrk[qi] = v->mrank[idx[qi]];
        }
        free(idx);
        free(v->ml); free(v->mr); free(v->mid); free(v->mrank);
        v->ml = nml; v->mr = nmr; v->mid = nmid; v->mrank = nmrk;
        /* 释放原始串 */
        for (i = 0; i < (int)v->n_merges; i++) { free(v->mls[i]); free(v->mrs[i]); }
        free(v->mls); free(v->mrs);
        v->mls = NULL; v->mrs = NULL;
    }
    /* merges 存在且无 scores -> byte-level BPE(qwen2), 预计算字节表 */
    if (v->n_merges > 0 && v->n_scores == 0) {
        v->byte_level = 1;
        build_byte_ids(v);
    }
    return 0;
}

void vocab_free(Vocab* v)
{
    int i;
    if (v->pieces) {
        for (i = 0; i < v->n; i++) free(v->pieces[i]);
        free(v->pieces);
    }
    free(v->order);
    free(v->sorted);
    free(v->scores);
    free(v->ml);
    free(v->mr);
    free(v->mid);
    free(v->mrank);
    if (v->mls) {
        int mi;
        for (mi = 0; mi < (int)v->n_merges; mi++) { free(v->mls[mi]); free(v->mrs[mi]); }
        free(v->mls);
        free(v->mrs);
    }
    free(v->chat_template);
    memset(v, 0, sizeof(*v));
}

int vocab_id(Vocab* v, const char* piece)
{
    int i;
    for (i = 0; i < v->n; i++) {
        if (strcmp(v->pieces[i], piece) == 0) return i;
    }
    return -1;
}

/* greedy fallback (dummy models, sp vocab without merges) */
static int vocab_encode_greedy(Vocab* v, const char* text, uint32_t* ids, int max)
{
    int nout = 0;
    const char* p = text;
    while (*p && nout < max) {
        int best = -1;
        size_t bestlen = 0;
        int i;
        for (i = 0; i < v->n; i++) {
            const char* pc = v->pieces[i];
            size_t l = strlen(pc);
            if (l == 0 || l < bestlen) continue;
            if (strncmp(p, pc, l) == 0) { best = i; bestlen = l; }
        }
        if (best < 0) {
            char bp[16];
            uint8_t c = (uint8_t)*p;
            snprintf(bp, sizeof(bp), "<0x%02X>", c);
            int id = vocab_id(v, bp);
            if (id >= 0) {
                ids[nout++] = (uint32_t)id;
            } else if (v->unk >= 0) {
                ids[nout++] = (uint32_t)v->unk;
            }
            p += 1;
        } else {
            ids[nout++] = (uint32_t)best;
            p += bestlen;
        }
    }
    return nout;
}

static int utf8_clen(unsigned char c);

/* ---- GPT-2/tiktoken byte->Unicode 映射(qwen2 等 byte-level BPE) ----
 * 特殊字节 0x00-0x20, 0x7F-0x9F, 0xA0, 0xAD -> U+0100..U+0143(依序)
 * 其余字节保持 Latin-1(0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF)
 */
static const unsigned char tiktoken_special[256] = {
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,0,0,0,0,0,0,0,0,0,0,0,0,
    1,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};

/* 把原始字节 b 写成 tiktoken 映射后的 UTF-8 串, 返回长度 */
static int gpt2_byte_str(unsigned int b, char out[4])
{
    unsigned int cp;
    if (tiktoken_special[b]) {
        /* 特殊字节按序: 0x00-0x20(33 个), 0x7F-0x9F(33 个), 0xA0, 0xAD */
        unsigned int idx;
        if (b <= 0x20) idx = b;
        else if (b <= 0x9F) idx = 33 + (b - 0x7F);
        else if (b == 0xA0) idx = 66;
        else idx = 67;
        cp = 0x100 + idx;
    } else {
        cp = b;
    }
    if (cp < 0x80) {
        out[0] = (char)cp; return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
}

/* 预计算 256 字节的 token id(byte_level 时在 vocab_load 末尾调用) */
static void build_byte_ids(Vocab* v)
{
    int b;
    for (b = 0; b < 256; b++) {
        char mapped[4];
        int len = gpt2_byte_str((unsigned int)b, mapped);
        int id = -1;
        if (vocab_bsearch_sorted(v, v->sorted, mapped, (size_t)len, &id) == 0) {
            v->byte_ids[b] = id;
        } else {
            v->byte_ids[b] = -1;
        }
    }
}

/* qwen 特殊 token: 编码前整体匹配(含 thinking 启停, 切成 BPE 会让模型看到乱码) */
static const char* k_qwen_specials[] = {
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<think>", "</think>",
    "<|extra_0|>", "<|extra_1|>", "<|extra_2|>", "<|extra_3|>",
    NULL
};

static int special_token_id(const Vocab* v, const char* text)
{
    size_t i;
    int best_id = -1;
    size_t best_len = 0;
    for (i = 0; k_qwen_specials[i]; i++) {
        size_t n = strlen(k_qwen_specials[i]);
        if (n > best_len && strncmp(text, k_qwen_specials[i], n) == 0) {
            int id = -1;
            if (vocab_bsearch_sorted(v, v->sorted, k_qwen_specials[i], n, &id) == 0) {
                best_id = id;
                best_len = n;
            }
        }
    }
    return best_id;
}

/* byte-level BPE encode using tokenizer.ggml.merges ranks (qwen2 etc.) */
static int vocab_encode_merges(Vocab* v, const char* text, uint32_t* ids, int max)
{
    if (v->n_merges == 0) return 0;

    size_t tlen = strlen(text);
    size_t cap = tlen * 2 + 8;
    uint32_t* syms = (uint32_t*)ymalloc((cap + 1) * 4);
    uint32_t ns = 0;
    size_t i = 0;

    while (i < tlen) {
        /* 特殊 token(<|im_start|> 等)整体匹配 */
        int sid = special_token_id(v, text + i);
        if (sid >= 0) {
            const char* sp = NULL;
            size_t k, nsp = 0;
            for (k = 0; k_qwen_specials[k]; k++) {
                size_t n = strlen(k_qwen_specials[k]);
                if (n > nsp && strncmp(text + i, k_qwen_specials[k], n) == 0) {
                    int id = -1;
                    if (vocab_bsearch_sorted(v, v->sorted, k_qwen_specials[k], n, &id) == 0 && id == sid) {
                        sp = k_qwen_specials[k];
                        nsp = n;
                    }
                }
            }
            if (!sp) { i += 1; continue; }
            if (ns < (uint32_t)max) syms[ns++] = (uint32_t)sid;
            i += nsp;
            continue;
        }
        /* 多字节 UTF-8 直接命中 */
        int clen = utf8_clen((unsigned char)text[i]);
        if (i + (size_t)clen > tlen) clen = (int)(tlen - i);
        int id = -1;
        if (clen > 1 && vocab_bsearch_sorted(v, v->sorted, text + i, (size_t)clen, &id) == 0) {
            if (ns < (uint32_t)max) syms[ns++] = (uint32_t)id;
            i += (size_t)clen;
            continue;
        }
        /* tiktoken 字节映射 */
        if (v->byte_ids[(unsigned char)text[i]] >= 0) {
            if (ns < (uint32_t)max) syms[ns++] = (uint32_t)v->byte_ids[(unsigned char)text[i]];
        } else if (v->unk >= 0) {
            if (ns < (uint32_t)max) syms[ns++] = (uint32_t)v->unk;
        }
        i += 1;
    }

    /* merge loop: repeatedly apply the lowest-rank merge among adjacent pairs */
    for (;;) {
        int best_i = -1;
        uint32_t best_out = 0;
        uint32_t best_rank = 0xFFFFFFFFu;
        uint32_t j;
        for (j = 0; j + 1 < ns; j++) {
            uint32_t a = syms[j], b = syms[j + 1];
            uint32_t lo = 0, hi = v->n_merges;
            while (lo < hi) {
                uint32_t mid = (lo + hi) / 2;
                if (v->ml[mid] < a || (v->ml[mid] == a && v->mr[mid] < b)) lo = mid + 1;
                else hi = mid;
            }
            if (lo < v->n_merges && v->ml[lo] == a && v->mr[lo] == b) {
                uint32_t rk = v->mrank[lo];
                if (rk < best_rank) { best_rank = rk; best_i = (int)j; best_out = v->mid[lo]; }
            }
        }
        if (best_i < 0 || ns >= (uint32_t)max) break;
        syms[best_i] = best_out;
        memmove(syms + best_i + 1, syms + best_i + 2, (size_t)(ns - best_i - 2) * 4);
        ns--;
    }

    uint32_t nout = ns < (uint32_t)max ? ns : (uint32_t)max;
    memcpy(ids, syms, (size_t)nout * 4);
    free(syms);
    return (int)nout;
}

/* sentencepiece-style BPE encode (llama/tinyllama etc.) */
static int utf8_clen(unsigned char c)
{
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}

int vocab_encode(Vocab* v, const char* text, uint32_t* ids, int max)
{
    if (v->n_scores == 0 && v->n_merges > 0) {
        return vocab_encode_merges(v, text, ids, max);
    }
    if (v->n_scores == 0) return vocab_encode_greedy(v, text, ids, max);

    /* 1. build normalized text: leading ▁, spaces -> ▁ (U+2581 = E2 96 81) */
    size_t tlen = strlen(text);
    size_t cap = tlen * 3 + 8;
    char* norm = (char*)ymalloc(cap);
    size_t nn = 0;
    norm[nn++] = (char)0xE2;
    norm[nn++] = (char)0x96;
    norm[nn++] = (char)0x81;
    size_t i;
    for (i = 0; i < tlen; i++) {
        if (text[i] == ' ') {
            norm[nn++] = (char)0xE2;
            norm[nn++] = (char)0x96;
            norm[nn++] = (char)0x81;
        } else {
            norm[nn++] = text[i];
        }
    }
    norm[nn] = 0;

    /* 2. initial symbols: whole UTF-8 char if in vocab, else byte tokens */
    uint32_t* syms = (uint32_t*)ymalloc(((size_t)nn + 1) * 4);
    uint32_t ns = 0;
    i = 0;
    while (i < nn) {
        int clen = utf8_clen((unsigned char)norm[i]);
        if (i + (size_t)clen > nn) clen = (int)(nn - i);
        int id = -1;
        if (vocab_bsearch_sorted(v, v->sorted, norm + i, (size_t)clen, &id) == 0) {
            syms[ns++] = (uint32_t)id;
            i += (size_t)clen;
            continue;
        }
        /* byte fallback: one symbol per byte */
        int b;
        for (b = 0; b < clen; b++) {
            char bt[16];
            snprintf(bt, sizeof(bt), "<0x%02X>", (unsigned char)norm[i + b]);
            if (vocab_bsearch_sorted(v, v->sorted, bt, strlen(bt), &id) == 0) {
                syms[ns++] = (uint32_t)id;
            } else if (v->unk >= 0) {
                syms[ns++] = (uint32_t)v->unk;
            }
        }
        i += (size_t)clen;
    }

    /* 3. merge loop: pick adjacent pair whose concatenation exists in vocab
       with the highest score; merge it (like picolm's bpe_merge) */
    char merged[512];
    for (;;) {
        float best_score = -1e30f;
        int best_i = -1;
        uint32_t best_out = 0;
        uint32_t j;
        for (j = 0; j + 1 < ns; j++) {
            const char* s1 = v->pieces[syms[j]];
            const char* s2 = v->pieces[syms[j + 1]];
            size_t l1 = strlen(s1);
            size_t l2 = strlen(s2);
            if (l1 + l2 >= sizeof(merged)) continue;
            memcpy(merged, s1, l1);
            memcpy(merged + l1, s2, l2);
            merged[l1 + l2] = 0;
            int mid;
            if (vocab_bsearch_sorted(v, v->sorted, merged, l1 + l2, &mid) == 0) {
                float sc = mid < (int)v->n_scores ? v->scores[mid] : 0.0f;
                if (sc > best_score) {
                    best_score = sc;
                    best_i = (int)j;
                    best_out = (uint32_t)mid;
                }
            }
        }
        if (best_i < 0 || ns >= (uint32_t)max) break;
        syms[best_i] = best_out;
        memmove(syms + best_i + 1, syms + best_i + 2, (size_t)(ns - best_i - 2) * 4);
        ns--;
    }

    uint32_t nout = ns < (uint32_t)max ? ns : (uint32_t)max;
    memcpy(ids, syms, (size_t)nout * 4);
    free(syms);
    free(norm);
    return (int)nout;
}

int vocab_decode(Vocab* v, const uint32_t* ids, int n, char* out, int max)
{
    int o = 0;
    int i;
    for (i = 0; i < n; i++) {
        uint32_t id = ids[i];
        if (id >= (uint32_t)v->n) continue;
        const char* pc = v->pieces[id];
        if (strcmp(pc, "<s>") == 0 || strcmp(pc, "</s>") == 0 ||
            strcmp(pc, "<bos>") == 0 || strcmp(pc, "<eos>") == 0 ||
            strcmp(pc, "<unk>") == 0 || strcmp(pc, "<pad>") == 0)
            continue;
        if (strncmp(pc, "<0x", 3) == 0 && strlen(pc) == 6 && pc[5] == '>') {
            unsigned int b;
            if (sscanf(pc + 3, "%2x", &b) == 1 && o < max) out[o++] = (char)b;
            continue;
        }
        size_t l = strlen(pc);
        if (v->byte_level) {
            /* tiktoken 反向映射: 特殊字节 U+0100..U+0143 -> 原始字节;
               Latin-1 非特殊(U+00A1-AC, U+00AE-FF)-> 原始字节 */
            size_t k = 0;
            while (k < l && o < max - 1) {
                unsigned char c = (unsigned char)pc[k];
                if (c >= 0xC2 && c < 0xE0 && k + 1 < l) {
                    unsigned int cp = ((unsigned int)(c & 0x1F) << 6) | ((unsigned int)(unsigned char)pc[k + 1] & 0x3F);
                    if (cp >= 0x0100 && cp <= 0x0143) {
                        unsigned int ob;
                        if (cp <= 0x0120) ob = cp - 0x0100;
                        else if (cp <= 0x0141) ob = 0x7F + (cp - 0x0121);
                        else if (cp == 0x0142) ob = 0xA0;
                        else ob = 0xAD;
                        out[o++] = (char)ob;
                        k += 2;
                        continue;
                    }
                    if ((cp >= 0x00A1 && cp <= 0x00AC) || cp >= 0x00AE) {
                        out[o++] = (char)cp;
                        k += 2;
                        continue;
                    }
                    out[o++] = (char)c;
                    k += 1;
                    continue;
                }
                out[o++] = (char)c;
                k += 1;
            }
            continue;
        }
        if (pc[0] == (char)0xe2 && pc[1] == (char)0x96 && pc[2] == (char)0x81) {
            if (o < max) out[o++] = ' ';
            pc += 3;
            l -= 3;
        }
        if (o + (int)l >= max) l = (size_t)(max - o);
        memcpy(out + o, pc, l);
        o += (int)l;
    }
    out[o] = 0;
    return o;
}

int vocab_has_template(Vocab* v)
{
    return v->chat_template && v->chat_template[0] != 0;
}

/* ---- simplified jinja2 chat-template renderer ----
 *
 * Supports the message-loop subset used by llama/qwen family templates:
 *
 *   {% for message in messages %}
 *   {% if message['role'] == 'user' %}
 *   {{ '<|user|>\n' + message['content'] + eos_token }}
 *   {% elif message['role'] == 'system' %} ...
 *   {% elif message['role'] == 'assistant' %} ...
 *   {% endif %}
 *   {% if loop.last and add_generation_prompt %}
 *   {{ '<|assistant|>' }}
 *   {% endif %}
 *   {% endfor %}
 *
 * Expressions evaluated: string literals ('...'), message['role'],
 * message['content'], eos_token, bos_token, + concatenation, \n escapes.
 */

typedef struct {
    const char* role;
    const char* content;
} ChatMsg;

/* evaluate a {{ ... }} expression into out, returns length */
static int chat_eval_expr(const char* e, const ChatMsg* msg, int is_last, int add_gen,
                          int eos_id, int bos_id, const Vocab* v, char* out, int max)
{
    int o = 0;
    const char* p = e;
    const char* eos_tok = eos_id >= 0 && eos_id < v->n ? v->pieces[eos_id] : "";
    const char* bos_tok = bos_id >= 0 && bos_id < v->n ? v->pieces[bos_id] : "";
    while (*p && o < max - 1) {
        if (*p == '\'') {
            /* string literal */
            p++;
            while (*p && *p != '\'') {
                if (*p == '\\' && p[1] == 'n') { out[o++] = '\n'; p += 2; }
                else if (*p == '\\' && p[1] == 't') { out[o++] = '\t'; p += 2; }
                else out[o++] = *p++;
            }
            if (*p == '\'') p++;
        } else if (strncmp(p, "message.role", 12) == 0) {
            const char* r = msg ? msg->role : "";
            while (*r && o < max - 1) out[o++] = *r++;
            p += 12;
        } else if (strncmp(p, "message.content", 15) == 0) {
            const char* c = msg ? msg->content : "";
            while (*c && o < max - 1) out[o++] = *c++;
            p += 15;
        } else if (strncmp(p, "message['role']", 15) == 0) {
            const char* r = msg ? msg->role : "";
            while (*r && o < max - 1) out[o++] = *r++;
            p += 15;
        } else if (strncmp(p, "message['content']", 18) == 0) {
            const char* c = msg ? msg->content : "";
            while (*c && o < max - 1) out[o++] = *c++;
            p += 18;
        } else if (strncmp(p, "eos_token", 9) == 0) {
            const char* s = eos_tok;
            while (*s && o < max - 1) out[o++] = *s++;
            p += 9;
        } else if (strncmp(p, "bos_token", 9) == 0) {
            const char* s = bos_tok;
            while (*s && o < max - 1) out[o++] = *s++;
            p += 9;
        } else if (strncmp(p, "loop.last", 9) == 0) {
            (void)is_last; (void)add_gen;
            p += 9;
        } else if (*p == '+') {
            p++;
        } else {
            p++;
        }
    }
    out[o] = 0;
    return o;
}

/* returns 1 if the condition is true */
static int chat_clause_true(const char* c, const ChatMsg* msg)
{
    while (*c == ' ' || *c == '(' || *c == ')') c++;
    if (strncmp(c, "True", 4) == 0) return 1;
    size_t l = strlen(c);
    if (strncmp(c, "add_generation_prompt", 21) == 0) return 1;
    if (strncmp(c, "loop.first", 10) == 0) return 1;      /* 单轮: 首条消息 */
    if (strncmp(c, "not loop.first", 14) == 0) return 0;  /* 单轮: 非首条 = 假 */
    if (strncmp(c, "tools", 5) == 0) return 0;
    if (strncmp(c, "message.content", 15) == 0) return msg && msg->content && msg->content[0] ? 1 : 0;
    {
        const char* eq = strstr(c, "==");
        if (eq) {
            const char* q1 = strchr(eq, '\'');
            const char* q2 = strchr(eq, '"');
            const char* q = q1 && (!q2 || q1 < q2) ? q1 : q2;
            if (q) {
                char endc = *q;
                const char* r = strchr(q + 1, endc);
                if (r) {
                    size_t rl = (size_t)(r - q - 1);
                    const char* role = msg ? msg->role : "";
                    if (rl == strlen(role) && strncmp(q + 1, role, rl) == 0) return 1;
                    return 0;
                }
            }
        }
    }
    return 0;
}

static int chat_eval_cond(const char* e, const ChatMsg* msg)
{
    /* 按 " or " 拆子句,任一为真即可(qwen2.5 模板) */
    const char* p = e;
    while (*p) {
        const char* orp = strstr(p, " or ");
        size_t cl = orp ? (size_t)(orp - p) : strlen(p);
        char tmp[512];
        if (cl >= sizeof(tmp)) cl = sizeof(tmp) - 1;
        memcpy(tmp, p, cl);
        tmp[cl] = 0;
        if (chat_clause_true(tmp, msg)) return 1;
        if (!orp) break;
        p = orp + 4;
    }
    return 0;
}

static int chat_append_ids(Vocab* v, const char* text, uint32_t* ids, int max, int* n)
{
    uint32_t tmp[4096];
    int ntmp = vocab_encode(v, text, tmp, 4096);
    int i;
    for (i = 0; i < ntmp && *n < max; i++) ids[(*n)++] = tmp[i];
    return 0;
}

/* HF-style: match whole special pieces first, SPM-encode the gaps. */
static int chat_spec_len_at(const char* p, const char** which)
{
    static const char* spec[] = {
        "<|tool_response>", "<tool_response|>",
        "<|tool_call>", "<tool_call|>",
        "<|turn>", "<turn|>",
        "<|tool>", "<tool|>",
        "<think>", "</think>",
        "<bos>", "<eos>", "<pad>", "<unk>",
        NULL
    };
    size_t best = 0;
    const char* w = NULL;
    int i;
    for (i = 0; spec[i]; i++) {
        size_t l = strlen(spec[i]);
        if (l > best && strncmp(p, spec[i], l) == 0) {
            best = l;
            w = spec[i];
        }
    }
    if (which) *which = w;
    return (int)best;
}

static int chat_append_ids_specials(Vocab* v, const char* text, uint32_t* ids, int max, int* n)
{
    const char* p = text;
    while (*p && *n < max) {
        const char* sp = NULL;
        int sl = chat_spec_len_at(p, &sp);
        if (sl > 0) {
            int id = vocab_id(v, sp);
            if (id >= 0 && *n < max) ids[(*n)++] = (uint32_t)id;
            p += sl;
            continue;
        }
        const char* e = p + 1;
        while (*e && chat_spec_len_at(e, NULL) == 0) e++;
        {
            size_t cl = (size_t)(e - p);
            char* chunk = (char*)ymalloc(cl + 1);
            memcpy(chunk, p, cl);
            chunk[cl] = 0;
            chat_append_ids(v, chunk, ids, max, n);
            free(chunk);
        }
        p = e;
    }
    return 0;
}

/* 检测简化渲染器不支持的 jinja 构造(macro/set/namespace/异常/反向循环等)。
 * 命中时若强行解析, macro 体会被当顶层语句执行, 输出 raise_exception 文本等乱码。 */
static int chat_template_unsupported(const char* tpl)
{
    if (!tpl) return 1;
    if (strstr(tpl, "macro") != NULL) return 1;
    if (strstr(tpl, "namespace(") != NULL) return 1;
    if (strstr(tpl, "raise_exception") != NULL) return 1;
    if (strstr(tpl, "[::-1]") != NULL) return 1;
    if (strstr(tpl, "loop.previtem") != NULL) return 1;
    if (strstr(tpl, "loop.nextitem") != NULL) return 1;
    return 0;
}

static int chat_vocab_has_token(const Vocab* v, const char* s)
{
    int id;
    return vocab_bsearch_sorted(v, v->sorted, s, strlen(s), &id) == 0;
}

/* 内置通用 im_start 聊天模板(qwen 家族标准格式), 供复杂模板回退使用。
 *   <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
 * vocab 无 im_start/im_end 时退化为 "role: content" 纯文本。 */
static void chat_render_generic(Vocab* v, const ChatMsg* msgs, int n_msgs,
                                uint32_t* ids, int max, int* n_out)
{
    int im = chat_vocab_has_token(v, "<|im_start|>") && chat_vocab_has_token(v, "<|im_end|>");
    int mi;
    for (mi = 0; mi < n_msgs && *n_out < max; mi++) {
        const char* role = msgs[mi].role && msgs[mi].role[0] ? msgs[mi].role : "user";
        if (im) {
            char head[520];
            snprintf(head, sizeof(head), "<|im_start|>%s\n", role);
            chat_append_ids(v, head, ids, max, n_out);
            chat_append_ids(v, msgs[mi].content ? msgs[mi].content : "", ids, max, n_out);
            chat_append_ids(v, "<|im_end|>\n", ids, max, n_out);
        } else {
            chat_append_ids(v, role, ids, max, n_out);
            chat_append_ids(v, ": ", ids, max, n_out);
            chat_append_ids(v, msgs[mi].content ? msgs[mi].content : "", ids, max, n_out);
            chat_append_ids(v, "\n", ids, max, n_out);
        }
    }
    if (im) {
        chat_append_ids(v, "<|im_start|>assistant\n", ids, max, n_out);
        /* Qwen3.5/3.8 thinking: 模板默认注入 <think>\\n, 否则模型离分布会吐乱码 */
        if (chat_vocab_has_token(v, "<think>"))
            chat_append_ids(v, "<think>\n", ids, max, n_out);
    } else
        chat_append_ids(v, "assistant: ", ids, max, n_out);
}

/* Gemma 4: <|turn>role\\n ... <turn|>\\n  <|turn>model\\n */
static int chat_is_gemma4(const Vocab* v)
{
    return chat_vocab_has_token(v, "<|turn>") && chat_vocab_has_token(v, "<turn|>");
}

static void chat_render_gemma4(Vocab* v, const ChatMsg* msgs, int n_msgs,
                               uint32_t* ids, int max, int* n_out)
{
    size_t need = 64;
    int mi;
    char* buf;
    size_t o = 0;
    for (mi = 0; mi < n_msgs; mi++) {
        const char* role = msgs[mi].role && msgs[mi].role[0] ? msgs[mi].role : "user";
        const char* c = msgs[mi].content ? msgs[mi].content : "";
        need += strlen(role) + strlen(c) + 32;
    }
    buf = (char*)ymalloc(need);
    buf[0] = 0;
    for (mi = 0; mi < n_msgs; mi++) {
        const char* role = msgs[mi].role && msgs[mi].role[0] ? msgs[mi].role : "user";
        const char* c = msgs[mi].content ? msgs[mi].content : "";
        int w;
        if (strcmp(role, "assistant") == 0) role = "model";
        w = snprintf(buf + o, need - o, "<|turn>%s\n%s<turn|>\n", role, c);
        if (w > 0) o += (size_t)w;
        if (o >= need) break;
    }
    if (o < need)
        snprintf(buf + o, need - o, "<|turn>model\n");
    chat_append_ids_specials(v, buf, ids, max, n_out);
    free(buf);
}

/* 多消息模板渲染: roles[i]/contents[i] 组成对话历史, 渲染完整模板。
 * 返回渲染出的 token 数; 无模板返回 -1。 */
int vocab_chat_ids_multi(Vocab* v, const char* const* roles, const char* const* contents,
                         int n_msgs, uint32_t* ids, int max, int add_bos)
{
    if (!vocab_has_template(v)) return -1;
    if (n_msgs < 1) return -1;

    ChatMsg* msgs = (ChatMsg*)ymalloc((size_t)n_msgs * sizeof(ChatMsg));
    int mi2;
    for (mi2 = 0; mi2 < n_msgs; mi2++) {
        msgs[mi2].role = roles ? roles[mi2] : "user";
        msgs[mi2].content = contents ? contents[mi2] : "";
    }

    int n_out = 0;
    if (add_bos && v->bos >= 0 && n_out < max) ids[n_out++] = (uint32_t)v->bos;

    if (chat_template_unsupported(v->chat_template)) {
        if (chat_is_gemma4(v))
            chat_render_gemma4(v, msgs, n_msgs, ids, max, &n_out);
        else
            chat_render_generic(v, msgs, n_msgs, ids, max, &n_out);
        free(msgs);
        return n_out;
    }

    /* parse template into statements by scanning {% %} / {{ }} blocks */
    enum { ST_FOR, ST_IF, ST_ELIF, ST_ENDIF, ST_END_FOR, ST_EXPR, ST_IF_LAST, ST_NONE };
    typedef struct {
        int kind;
        char cond[512];   /* for ST_IF/ST_ELIF: the condition expr */
        char expr[2048];  /* for ST_EXPR: the expression */
    } TStmt;
    TStmt stmts[128];
    int n_stmts = 0;

    {
        const char* p = v->chat_template;
        while (*p && n_stmts < 128) {
            if (p[0] == '{' && p[1] == '%') {
                /* control block: {% ... %} */
                const char* e = strstr(p + 2, "%}");
                if (!e) break;
                size_t bl = (size_t)(e - (p + 2));
                char tmp[512];
                if (bl >= sizeof(tmp)) bl = sizeof(tmp) - 1;
                memcpy(tmp, p + 2, bl);
                tmp[bl] = 0;
                char* t2 = tmp;
                while (*t2 == ' ' || *t2 == '\n' || *t2 == '\t') t2++;
                if (*t2 == '-') t2++;   /* {%- 空白控制语法 */
                while (*t2 == ' ' || *t2 == '\n' || *t2 == '\t') t2++;
                {
                    size_t tl = strlen(t2);
                    while (tl > 0 && (t2[tl - 1] == ' ' || t2[tl - 1] == '\n' || t2[tl - 1] == '\t' || t2[tl - 1] == '-')) t2[--tl] = 0;
                }
                if (strncmp(t2, "for", 3) == 0) {
                    stmts[n_stmts].kind = ST_FOR;
                    n_stmts++;
                } else if (strncmp(t2, "if loop.last", 12) == 0) {
                    stmts[n_stmts].kind = ST_IF_LAST;
                    n_stmts++;
                } else if (strncmp(t2, "if", 2) == 0) {
                    stmts[n_stmts].kind = ST_IF;
                    snprintf(stmts[n_stmts].cond, sizeof(stmts[n_stmts].cond), "%s", t2 + 2);
                    n_stmts++;
                } else if (strncmp(t2, "elif", 4) == 0) {
                    stmts[n_stmts].kind = ST_ELIF;
                    snprintf(stmts[n_stmts].cond, sizeof(stmts[n_stmts].cond), "%s", t2 + 4);
                    n_stmts++;
                } else if (strncmp(t2, "else", 4) == 0) {
                    stmts[n_stmts].kind = ST_ELIF;
                    snprintf(stmts[n_stmts].cond, sizeof(stmts[n_stmts].cond), "True");
                    n_stmts++;
                } else if (strncmp(t2, "endif", 5) == 0) {
                    stmts[n_stmts].kind = ST_ENDIF;
                    n_stmts++;
                } else if (strncmp(t2, "endfor", 6) == 0) {
                    stmts[n_stmts].kind = ST_END_FOR;
                    n_stmts++;
                }
                p = e + 2;
            } else if (p[0] == '{' && p[1] == '{') {
                /* expression block: {{ ... }} (may span newlines) */
                const char* e = strstr(p + 2, "}}");
                if (!e) break;
                size_t bl = (size_t)(e - (p + 2));
                char tmp[2048];
                if (bl >= sizeof(tmp)) bl = sizeof(tmp) - 1;
                memcpy(tmp, p + 2, bl);
                tmp[bl] = 0;
                stmts[n_stmts].kind = ST_EXPR;
                snprintf(stmts[n_stmts].expr, sizeof(stmts[n_stmts].expr), "%s", tmp);
                n_stmts++;
                p = e + 2;
            } else {
                p++;
            }
        }
    }
#ifdef CHAT_DEBUG
    {
        int di;
        fprintf(stderr, "CHAT n_stmts=%d\n", n_stmts);
        for (di = 0; di < n_stmts; di++) {
            fprintf(stderr, "  stmt%d kind=%d cond=[%s] expr=[%s]\n", di, stmts[di].kind,
                    stmts[di].cond, stmts[di].expr);
        }
    }
#endif

    /* execute statements, iterating messages */
    char out[4096];
    int mi;
    for (mi = 0; mi < n_msgs; mi++) {
        int is_last = (mi == n_msgs - 1);
        /* if 栈: 每层 1 = 本链未匹配(跳过), 2 = 已匹配(渲染),
         * 3 = 位于已跳过区域内的链(永久跳过, else/elif 不得复活) */
        int stk[16], sp = 0;
        int si;
        for (si = 0; si < n_stmts && n_out < max; si++) {
            TStmt* st = &stmts[si];
            switch (st->kind) {
            case ST_FOR:
                break;
            case ST_IF:
                if (sp > 0 && stk[sp - 1] == 1)
                    stk[sp++] = 3;      /* 未匹配分支内的嵌套 if */
                else if (sp > 0 && stk[sp - 1] == 3)
                    stk[sp++] = 3;      /* 跳过区域内: 层层跳过 */
                else
                    stk[sp++] = chat_eval_cond(st->cond, &msgs[mi]) ? 2 : 1;
                break;
            case ST_ELIF:
                if (sp > 0) {
                    if (stk[sp - 1] == 1) {
                        stk[sp - 1] = chat_eval_cond(st->cond, &msgs[mi]) ? 2 : 1;
                    } else if (stk[sp - 1] == 2) {
                        stk[sp - 1] = 1; /* 本链已匹配, 跳过剩余 elif */
                    }
                    /* 3: 保持跳过 */
                }
                break;
            case ST_ENDIF:
                if (sp > 0) sp--;
                break;
            case ST_IF_LAST:
                if (sp > 0 && stk[sp - 1] == 1)
                    stk[sp++] = 3;
                else if (sp > 0 && stk[sp - 1] == 3)
                    stk[sp++] = 3;
                else
                    stk[sp++] = is_last ? 2 : 1;
                break;
            case ST_END_FOR:
                break;
            case ST_EXPR:
                if (sp == 0 || stk[sp - 1] == 2) {
                    chat_eval_expr(st->expr, &msgs[mi], is_last, 1, v->eos, v->bos, v, out, sizeof(out));
                    chat_append_ids(v, out, ids, max, &n_out);
                }
                break;
            default:
                break;
            }
        }
        if (!is_last) continue;
    }
    free(msgs);
    return n_out;
}

/* single-turn 兼容包装 */
int vocab_chat_ids(Vocab* v, const char* user_msg, uint32_t* ids, int max, int add_bos)
{
    const char* roles[1] = {"user"};
    const char* contents[1] = {user_msg};
    return vocab_chat_ids_multi(v, roles, contents, 1, ids, max, add_bos);
}
