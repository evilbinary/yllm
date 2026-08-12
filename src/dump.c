/* llfdump: LLF / GGUF / Safetensors 模型文件内容查看器
 *
 *   build/llfdump <file>         自动按魔数识别格式并 dump
 *   build/llfdump <file> -v      详细模式(逐张量)
 */
#include "yllm.h"
#include "llf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_verbose = 0;
static const char* g_fname;

/* ---------- 工具 ---------- */
static uint32_t bswap32(uint32_t v)
{
    return ((v & 0xFFu) << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
}
static uint64_t bswap64(uint64_t v)
{
    return ((uint64_t)bswap32((uint32_t)v) << 32) | bswap32((uint32_t)(v >> 32));
}
static double fmtsize(uint64_t n, char* buf, size_t buflen)
{
    double v = (double)n;
    const char* u[] = { "B", "KB", "MB", "GB", "TB" };
    int i = 0;
    while (v >= 1024.0 && i < 4) { v /= 1024.0; i++; }
    snprintf(buf, buflen, "%.2f %s", v, u[i]);
    return v;
}

/* ---------- LLF ---------- */
static const char* llf_dtype_name_c(uint32_t dt)
{
    switch (dt) {
    case DT_F16: return "f16";
    case DT_F32: return "f32";
    case DT_BF16: return "bf16";
    case DT_Q4K: return "q4_k";
    case DT_Q6K: return "q6_k";
    case DT_IQ4XS: return "iq4_xs";
    default: return "?";
    }
}
static const char* llf_arch_name_c(uint32_t a)
{
    switch (a) {
    case ARCH_LLAMA: return "llama";
    case ARCH_QWEN: return "qwen2";
    default: return "?";
    }
}

static int dump_llf(const uint8_t* data, uint64_t fsize)
{
    if (fsize < LLF_HEADER_SIZE) { fprintf(stderr, "llfdump: file too small for LLF header\n"); return 1; }
    const LlfHeader* h = (const LlfHeader*)data;
    float eps, theta;
    memcpy(&eps, &h->norm_eps_bits, 4);
    memcpy(&theta, &h->rope_theta_bits, 4);
    printf("== LLF 文件: %s ==\n", g_fname);
    printf("magic=%.8s version=%u arch=%s\n", h->magic, h->version, llf_arch_name_c(h->arch));
    printf("blocks=%u vocab=%u hidden=%u heads=%u kv_heads=%u head_dim=%u\n",
           h->n_blocks, h->vocab, h->hidden, h->n_heads, h->n_kv_heads, h->head_dim);
    printf("max_seq=%u dtype=%s norm_eps=%.3g rope_theta=%.6g\n",
           h->max_seq, llf_dtype_name_c(h->dtype), eps, theta);
    printf("file_size=%llu (%.2f MB)\n", (unsigned long long)h->file_size, (double)h->file_size / 1048576.0);

    uint32_t n_layers = h->n_blocks + 3;
    uint64_t dir_off = LLF_HEADER_SIZE;
    uint64_t need = dir_off + (uint64_t)n_layers * LLF_DIR_ENTRY_SIZE +
                    (uint64_t)n_layers * BLOCK_TENSORS * LLF_TENSOR_META_SIZE;
    if (need > fsize) { fprintf(stderr, "llfdump: LLF metadata section out of range\n"); return 1; }
    const LlfLayerDir* dir = (const LlfLayerDir*)(data + dir_off);
    const LlfTensorMeta* metas = (const LlfTensorMeta*)(data + need - (uint64_t)n_layers * BLOCK_TENSORS * LLF_TENSOR_META_SIZE);
    char b1[32], b2[32];
    uint32_t i;
    for (i = 0; i < n_layers; i++) {
        fmtsize(dir[i].size, b1, sizeof(b1));
        const char* lname;
        if (i == 0) lname = "embed";
        else if (i <= h->n_blocks) lname = "transformer block";
        else if (i == h->n_blocks + 1) lname = "final norm";
        else lname = "output (lm_head)";
        if (i >= 1 && i <= h->n_blocks) {
            printf("layer %2u [%s %u]  off=%-12llu size=%-10s n_tensors=%u\n",
                   i, lname, i - 1, (unsigned long long)dir[i].offset, b1, dir[i].n_tensors);
        } else {
            printf("layer %2u [%s]      off=%-12llu size=%-10s n_tensors=%u\n",
                   i, lname, (unsigned long long)dir[i].offset, b1, dir[i].n_tensors);
        }
        if (!g_verbose) continue;
        uint32_t j;
        for (j = 0; j < BLOCK_TENSORS; j++) {
            const LlfTensorMeta* tm = &metas[(size_t)i * BLOCK_TENSORS + j];
            if (tm->size == 0 && tm->offset == 0 && tm->name[0] == 0) continue;
            fmtsize(tm->size, b2, sizeof(b2));
            printf("  [%2u] %-24s %-7s dims=%u shape=(%u,%u,%u,%u) off=%-12llu size=%-10s\n",
                   j, tm->name, llf_dtype_name_c(tm->dtype), tm->ndim,
                   tm->shape[0], tm->shape[1], tm->shape[2], tm->shape[3],
                   (unsigned long long)tm->offset, b2);
        }
    }
    return 0;
}

/* ---------- GGUF ---------- */
typedef struct {
    const uint8_t* p;
    const uint8_t* end;
    int err;
    int be;
} GB;

static uint32_t gb_u32(GB* b)
{
    uint32_t v;
    if (b->p + 4 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 4);
    if (b->be) v = bswap32(v);
    b->p += 4;
    return v;
}
static uint64_t gb_u64(GB* b)
{
    uint64_t v;
    if (b->p + 8 > b->end) { b->err = 1; return 0; }
    memcpy(&v, b->p, 8);
    if (b->be) v = bswap64(v);
    b->p += 8;
    return v;
}
static char* gb_str(GB* b)
{
    uint64_t len = gb_u64(b);
    if (b->err || b->p + len > b->end) { b->err = 1; return NULL; }
    char* s = (char*)ymalloc((size_t)len + 1);
    memcpy(s, b->p, (size_t)len);
    s[len] = 0;
    b->p += len;
    return s;
}

static const char* gguf_typename(uint32_t t)
{
    switch (t) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "BF16";
    case 3: return "F8_E4M3";
    case 4: return "F8_E5M2";
    case 7: return "Q8_0";
    case 8: return "Q8_1";
    case 9: return "Q2_K";
    case 10: return "Q3_K";
    case 11: return "Q4_0";
    case 12: return "Q4_1";
    case 13: return "Q4_K";
    case 14: return "Q5_K";
    case 15: return "Q6_K";
    case 16: return "IQ4_NL";
    case 17: return "IQ4_XS";
    case 18: return "IQ3_XXS";
    case 19: return "IQ1_S";
    case 20: return "IQ4_NL";
    default: return "?";
    }
}

static void dump_gguf_kv(GB* b, const char* key, uint32_t type)
{
    char vbuf[256];
    switch (type) {
    case 0: { uint8_t v = (uint8_t)*b->p; b->p += 1; snprintf(vbuf, sizeof(vbuf), "%u", v); break; }
    case 1: { int8_t v = (int8_t)*b->p; b->p += 1; snprintf(vbuf, sizeof(vbuf), "%d", v); break; }
    case 2: { uint16_t v = (uint16_t)gb_u32(b); snprintf(vbuf, sizeof(vbuf), "%u", v); break; }
    case 3: { int16_t v = (int16_t)gb_u32(b); snprintf(vbuf, sizeof(vbuf), "%d", v); break; }
    case 4: { uint32_t v = gb_u32(b); snprintf(vbuf, sizeof(vbuf), "%u", v); break; }
    case 5: { int32_t v = (int32_t)gb_u32(b); snprintf(vbuf, sizeof(vbuf), "%d", v); break; }
    case 6: { uint32_t v = gb_u32(b); float f; memcpy(&f, &v, 4); snprintf(vbuf, sizeof(vbuf), "%g", (double)f); break; }
    case 7: { uint8_t v = (uint8_t)*b->p; b->p += 1; snprintf(vbuf, sizeof(vbuf), "%s", v ? "true" : "false"); break; }
    case 8: {
        char* s = gb_str(b);
        snprintf(vbuf, sizeof(vbuf), "\"%s\"", s ? s : "?");
        free(s);
        break;
    }
    case 9: {
        uint32_t at = gb_u32(b);
        uint64_t n = gb_u64(b);
        snprintf(vbuf, sizeof(vbuf), "array[%s]x%llu", gguf_typename(at), (unsigned long long)n);
        uint64_t i;
        for (i = 0; i < n && !b->err; i++) {
            switch (at) {
            case 0: case 1: case 7: b->p += 1; break;
            case 2: case 3: b->p += 2; break;
            case 4: case 5: case 6: b->p += 4; break;
            case 10: case 11: case 12: b->p += 8; break;
            case 8: { char* s = gb_str(b); free(s); break; }
            default: b->err = 1;
            }
        }
        break;
    }
    case 10: { uint64_t v = gb_u64(b); snprintf(vbuf, sizeof(vbuf), "%llu", (unsigned long long)v); break; }
    case 11: { int64_t v = (int64_t)gb_u64(b); snprintf(vbuf, sizeof(vbuf), "%lld", (long long)v); break; }
    case 12: { uint64_t v = gb_u64(b); double f; memcpy(&f, &v, 8); snprintf(vbuf, sizeof(vbuf), "%g", f); break; }
    default: snprintf(vbuf, sizeof(vbuf), "type?%u", type); b->err = 1;
    }
    if (!b->err) printf("  %-42s = %s\n", key, vbuf);
}

static int dump_gguf(const uint8_t* data, uint64_t fsize)
{
    if (fsize < 24) return 1;
    GB b;
    b.p = data + 4;
    b.end = data + fsize;
    b.err = 0;
    b.be = 0;
    uint32_t ver = gb_u32(&b);
    if ((ver & 0xFFFF) == 0) { b.be = 1; ver = gb_u32(&b); }
    if (ver < 1 || ver > 3) { fprintf(stderr, "llfdump: unsupported gguf version %u\n", ver); return 1; }
    uint64_t n_tensors, n_kv;
    if (ver == 1) { n_tensors = gb_u32(&b); n_kv = gb_u32(&b); }
    else { n_tensors = gb_u64(&b); n_kv = gb_u64(&b); }
    printf("== GGUF 文件: %s ==\n", g_fname);
    printf("version=%u%s tensors=%llu kv=%llu\n", ver, b.be ? " (big-endian)" : "",
           (unsigned long long)n_tensors, (unsigned long long)n_kv);

    uint64_t i;
    for (i = 0; i < n_kv && !b.err; i++) {
        char* key = gb_str(&b);
        if (!key) break;
        uint32_t type = gb_u32(&b);
        dump_gguf_kv(&b, key, type);
        free(key);
    }
    if (b.err) { fprintf(stderr, "llfdump: bad gguf kv section\n"); return 1; }
    uint64_t meta_end = (uint64_t)(b.p - data);

    printf("\nTensors:\n");
    for (i = 0; i < n_tensors && !b.err; i++) {
        char* name = gb_str(&b);
        if (!name) break;
        uint32_t ndims = gb_u32(&b);
        uint64_t nelem = 1;
        uint32_t d;
        uint64_t dims[4] = { 0, 0, 0, 0 };
        for (d = 0; d < ndims && d < 4; d++) {
            dims[d] = gb_u64(&b);
            nelem *= dims[d];
        }
        uint32_t gtype = gb_u32(&b);
        uint64_t off = gb_u64(&b);
        char nb[32], sz[32];
        uint64_t nbytes = 0;
        if (gtype == 0) nbytes = nelem * 4;
        else if (gtype == 1) nbytes = nelem * 2;
        else if (gtype == 2) nbytes = nelem * 2;
        else if (gtype == 13) nbytes = nelem / 256 * 144;
        else if (gtype == 15) nbytes = nelem / 256 * 210;
        else if (gtype == 10) nbytes = nelem / 256 * 144;
        else if (gtype == 11) nbytes = nelem / 256 * 110;
        else if (gtype == 9) nbytes = nelem / 256 * 16;
        else nbytes = nelem * 1;
        fmtsize(nbytes, nb, sizeof(nb));
        printf("  %-44s %-9s nelem=%-12llu off=%-12llu %s\n", name, gguf_typename(gtype),
               (unsigned long long)nelem, (unsigned long long)off, nb);
        if (g_verbose) {
            printf("      dims=[");
            for (d = 0; d < ndims; d++) printf("%s%llu", d ? ", " : " ", (unsigned long long)dims[d]);
            printf(" ]\n");
        }
        free(name);
    }
    if (b.err) { fprintf(stderr, "llfdump: bad gguf tensor section\n"); return 1; }
    printf("\nmetadata section: %llu bytes\n", (unsigned long long)meta_end);
    return 0;
}

/* ---------- Safetensors ---------- */
static int dump_safetensors(const uint8_t* data, uint64_t fsize)
{
    if (fsize < 8) return 1;
    uint64_t hlen;
    memcpy(&hlen, data, 8);
    if (hlen + 8 > fsize) { fprintf(stderr, "llfdump: bad safetensors header length\n"); return 1; }
    const char* hs = (const char*)data + 8;
    printf("== Safetensors 文件: %s ==\n", g_fname);
    printf("header_len=%llu (%llu bytes)\n", (unsigned long long)hlen, (unsigned long long)hlen);

    /* 扫描 "name":{"dtype":"X","shape":[a,b,...],"data_offsets":[x,y]} */
    const char* p = hs;
    const char* end = hs + hlen;
    long count = 0;
    while (p < end) {
        const char* q = strchr(p, '"');
        if (!q) break;
        const char* q2 = strchr(q + 1, '"');
        if (!q2) break;
        char name[512];
        size_t nl = (size_t)(q2 - q - 1);
        if (nl >= sizeof(name)) nl = sizeof(name) - 1;
        memcpy(name, q + 1, nl);
        name[nl] = 0;
        const char* colon = strchr(q2, ':');
        if (!colon) break;
        const char* brace = strchr(colon, '{');
        if (!brace) break;
        const char* cend = strchr(brace, '}');
        if (!cend) break;
        /* 提取 dtype / shape / data_offsets */
        char dt[32] = "?";
        char shape[128] = "";
        char offs[64] = "";
        const char* s = brace;
        const char* se = cend;
        const char* dtk = strstr(s, "\"dtype\"");
        if (dtk && dtk < se) {
            const char* dv = strchr(dtk, ':');
            const char* dq = dv ? strchr(dv, '"') : NULL;
            if (dq) {
                const char* dq2 = strchr(dq + 1, '"');
                if (dq2) {
                    size_t dl = (size_t)(dq2 - dq - 1);
                    if (dl >= sizeof(dt)) dl = sizeof(dt) - 1;
                    memcpy(dt, dq + 1, dl);
                    dt[dl] = 0;
                }
            }
        }
        const char* shk = strstr(s, "\"shape\"");
        if (shk && shk < se) {
            const char* sv = strchr(shk, ':');
            const char* sb = sv ? strchr(sv, '[') : NULL;
            if (sb) {
                const char* sb2 = strchr(sb, ']');
                if (sb2) {
                    size_t sl = (size_t)(sb2 - sb - 1);
                    if (sl >= sizeof(shape)) sl = sizeof(shape) - 1;
                    memcpy(shape, sb + 1, sl);
                    shape[sl] = 0;
                }
            }
        }
        const char* dok = strstr(s, "\"data_offsets\"");
        if (dok && dok < se) {
            const char* dv2 = strchr(dok, ':');
            const char* db = dv2 ? strchr(dv2, '[') : NULL;
            if (db) {
                const char* db2 = strchr(db, ']');
                if (db2) {
                    size_t dl = (size_t)(db2 - db - 1);
                    if (dl >= sizeof(offs)) dl = sizeof(offs) - 1;
                    memcpy(offs, db + 1, dl);
                    offs[dl] = 0;
                }
            }
        }
        uint64_t start = 0, size = 0;
        sscanf(offs, "%llu, %llu", (unsigned long long*)&start, (unsigned long long*)&size);
        char nb[32];
        fmtsize(size, nb, sizeof(nb));
        printf("  %-44s %-8s shape=[%s] data_offsets=[%s] %s\n", name, dt, shape, offs, nb);
        count++;
        p = cend + 1;
    }
    printf("\ntensors: %ld\n", count);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: llfdump <file.llf|file.gguf|file.safetensors> [-v]\n");
        return 1;
    }
    g_fname = argv[1];
    if (argc > 2 && strcmp(argv[2], "-v") == 0) g_verbose = 1;

    uint64_t fsize = 0;
    if (yfile_size(g_fname, &fsize) != 0) { fprintf(stderr, "llfdump: cannot open %s\n", g_fname); return 1; }
    FILE* f = fopen(g_fname, "rb");
    if (!f) { fprintf(stderr, "llfdump: cannot open %s\n", g_fname); return 1; }
    uint8_t* data = (uint8_t*)ymalloc((size_t)fsize);
    if (fread(data, 1, (size_t)fsize, f) != fsize) { fclose(f); free(data); fprintf(stderr, "llfdump: read failed\n"); return 1; }
    fclose(f);

    int rc;
    if (fsize >= 8 && memcmp(data, "YLLMLLF1", 8) == 0) {
        rc = dump_llf(data, fsize);
    } else if (fsize >= 4 && memcmp(data, "GGUF", 4) == 0) {
        rc = dump_gguf(data, fsize);
    } else if (fsize >= 8) {
        rc = dump_safetensors(data, fsize);
    } else {
        fprintf(stderr, "llfdump: unknown format\n");
        rc = 1;
    }
    free(data);
    return rc;
}
