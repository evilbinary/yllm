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
                if (strcmp(pc, "<s>") == 0) v->bos = v->n;
                if (strcmp(pc, "</s>") == 0) v->eos = v->n;
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

static int parse_text(const char* path, Vocab* v)
{
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    char line[65536];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    int n = atoi(line);
    if (n <= 0 || n > 1000000) { fclose(f); return -1; }
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
            if (strcmp(pc, "<s>") == 0) v->bos = v->n;
            if (strcmp(pc, "</s>") == 0) v->eos = v->n;
        }
        v->n++;
    }
    if (v->n == 0) { fclose(f); return -1; }

    /* optional #SCORES# section: one float per piece */
    if (fgets(line, sizeof(line), f)) {
        size_t l = strlen(line);
        while (l > 0 && (line[l - 1] == '\n' || line[l - 1] == '\r')) line[--l] = 0;
        if (strcmp(line, "#SCORES#") == 0) {
            if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
            int ns = atoi(line);
            if (ns > 0 && ns <= 1000000) {
                v->scores = (float*)ymalloc((size_t)ns * sizeof(float));
                v->n_scores = (uint32_t)ns;
                uint32_t si = 0;
                while (si < v->n_scores && fgets(line, sizeof(line), f)) {
                    v->scores[si++] = (float)atof(line);
                }
            }
        }
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
    {
        char** tmp = (char**)ymalloc((size_t)v->n * sizeof(char*));
        for (i = 0; i < v->n; i++) tmp[i] = v->pieces[i];
        qsort(tmp, (size_t)v->n, sizeof(char*), order_compare);
        for (i = 0; i < v->n; i++) {
            int j;
            for (j = 0; j < v->n; j++) {
                if (strcmp(tmp[i], v->pieces[j]) == 0) v->order[i] = j;
            }
        }
        qsort(tmp, (size_t)v->n, sizeof(char*), dict_compare);
        for (i = 0; i < v->n; i++) {
            int j;
            for (j = 0; j < v->n; j++) {
                if (strcmp(tmp[i], v->pieces[j]) == 0) { v->sorted[i] = j; break; }
            }
        }
        free(tmp);
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
            if (l == 0 || l > bestlen) continue;
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
        if (strcmp(pc, "<s>") == 0 || strcmp(pc, "</s>") == 0) continue;
        if (strncmp(pc, "<0x", 3) == 0 && strlen(pc) == 6 && pc[5] == '>') {
            unsigned int b;
            if (sscanf(pc + 3, "%2x", &b) == 1 && o < max) out[o++] = (char)b;
            continue;
        }
        size_t l = strlen(pc);
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
