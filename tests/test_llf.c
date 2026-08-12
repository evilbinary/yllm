#include "yllm.h"
#include "convert.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); } \
} while (0)

/* ---- llf_emit layer layout: sizes must be positive and contiguous ---- *
 * Regression: lend was computed as relative while loff absolute, producing
 * huge negative dir[].size for every layer after layer 0 (resident ~17 PB,
 * engine read garbage). */
static void test_emit_layout(void)
{
    LlfHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, YLLM_MAGIC, 8);
    h.version = YLLM_VERSION;
    h.arch = ARCH_LLAMA;
    h.n_blocks = 2;
    h.hidden = 256;
    h.n_heads = 4;
    h.n_kv_heads = 1;
    h.head_dim = 64;
    h.max_seq = 64;
    h.vocab = 100;
    h.dtype = DT_Q4K;
    {
        float eps = 1e-5f, theta = 10000.0f;
        memcpy(&h.norm_eps_bits, &eps, 4);
        memcpy(&h.rope_theta_bits, &theta, 4);
    }

    /* items: layer0 emb, layer1..2 blocks (9 each), final norm, output */
    ConvItem items[22];
    int n = 0;
    int l;
    uint32_t i;
    memset(items, 0, sizeof(items));

    items[n].layer = 0; items[n].slot = 0;
    items[n].dtype = DT_Q4K; items[n].ndim = 2;
    items[n].shape[0] = 256; items[n].shape[1] = 100;
    items[n].nbytes = 100 * 256 / 256 * 144;
    snprintf(items[n].name, sizeof(items[n].name), "token_embd.weight");
    items[n].src = (const uint8_t*)calloc(1, items[n].nbytes ? items[n].nbytes : 1);
    n++;

    for (l = 1; l <= 2; l++) {
        static const int slots[9] = { 0,1,2,3,4,5,6,7,8 };
        static const uint32_t dims[9][2] = {
            { 256, 1 }, { 256, 256 }, { 256, 64 }, { 256, 64 },
            { 256, 256 }, { 256, 1 }, { 256, 512 }, { 256, 512 }, { 512, 256 }
        };
        int s;
        for (s = 0; s < 9; s++) {
            items[n].layer = (uint32_t)l;
            items[n].slot = (uint32_t)slots[s];
            items[n].dtype = DT_Q4K;
            items[n].ndim = 2;
            items[n].shape[0] = dims[s][0];
            items[n].shape[1] = dims[s][1];
            items[n].nbytes = (uint64_t)dims[s][0] * dims[s][1] / 256 * 144;
            snprintf(items[n].name, sizeof(items[n].name), "blk.%d.t%d", l - 1, s);
            items[n].src = (const uint8_t*)calloc(1, items[n].nbytes ? items[n].nbytes : 1);
            n++;
        }
    }
    items[n].layer = 3; items[n].slot = 0;
    items[n].dtype = DT_F32; items[n].ndim = 1;
    items[n].shape[0] = 256;
    items[n].nbytes = 256 * 4;
    snprintf(items[n].name, sizeof(items[n].name), "output_norm.weight");
    items[n].src = (const uint8_t*)calloc(1, items[n].nbytes);
    n++;

    items[n].layer = 4; items[n].slot = 0;
    items[n].dtype = DT_Q4K; items[n].ndim = 2;
    items[n].shape[0] = 256; items[n].shape[1] = 100;
    items[n].nbytes = 100 * 256 / 256 * 144;
    snprintf(items[n].name, sizeof(items[n].name), "output.weight");
    items[n].src = (const uint8_t*)calloc(1, items[n].nbytes);
    n++;

    char err[1024];
    const char* path = "build/test_emit.llf";
    CHECK(llf_emit(path, &h, items, n, err, sizeof(err)) == 0, "llf_emit succeeds");
    if (g_fail) {
        printf("  emit error: %s\n", err);
        for (i = 0; i < (uint32_t)n; i++) free((void*)items[i].src);
        return;
    }

    /* read back and verify layout */
    WMap map;
    CHECK(wmap_open(path, &map) == 0, "wmap_open");
    if (g_fail) {
        for (i = 0; i < (uint32_t)n; i++) free((void*)items[i].src);
        return;
    }
    LlModel m;
    CHECK(llf_read(&map, &m) == 0, "llf_read");
    if (g_fail) {
        wmap_close(&map);
        for (i = 0; i < (uint32_t)n; i++) free((void*)items[i].src);
        return;
    }
    CHECK(m.n_layers == 5, "5 layers");
    uint32_t li;
    uint64_t prev_end = 0;
    for (li = 0; li < m.n_layers; li++) {
        CHECK(m.dir[li].offset % LLF_ALIGN == 0, "layer offset aligned");
        CHECK(m.dir[li].size > 0, "layer size positive (was negative)");
        if (li > 0) {
            /* layer starts at or after previous end (may have alignment padding) */
            CHECK(m.dir[li].offset >= prev_end, "layers non-overlapping");
            CHECK(m.dir[li].offset < prev_end + LLF_ALIGN, "layers with small padding");
        }
        prev_end = m.dir[li].offset + m.dir[li].size;
        CHECK(m.dir[li].offset + m.dir[li].size <= map.size, "layer within file");
    }
    CHECK(m.dir[1].n_tensors == 9, "block layer has 9 tensors");
    /* tensor metas: slot offsets monotonic within layer */
    {
        const LlfTensorMeta* mt = &m.metas[m.base_idx[1]];
        uint32_t s;
        for (s = 1; s < 9; s++) {
            CHECK(mt[s].offset > mt[s - 1].offset, "tensor offsets increasing");
        }
    }
    /* verify no negative sizes in metas either */
    {
        uint32_t total = m.n_layers * BLOCK_TENSORS;
        for (li = 0; li < total; li++) {
            CHECK(m.metas[li].size < (uint64_t)1 << 62, "meta size sane");
        }
    }
    wmap_close(&map);
    remove(path);
    for (i = 0; i < (uint32_t)n; i++) free((void*)items[i].src);
}

/* ---- dummy model via convert_dummy round trip ---- */
static void test_dummy(void)
{
    char err[1024];
    const char* path = "build/test_dummy.llf";
    CHECK(convert_dummy(path, 2, 256, 4, 1, 100, 64, 42, err, sizeof(err)) == 0, "convert_dummy");
    if (g_fail) return;
    WMap map;
    CHECK(wmap_open(path, &map) == 0, "dummy wmap_open");
    if (g_fail) return;
    LlModel m;
    CHECK(llf_read(&map, &m) == 0, "dummy llf_read");
    if (g_fail) { wmap_close(&map); return; }
    CHECK(m.h.n_blocks == 2 && m.h.hidden == 256, "dummy config");
    CHECK(m.dir[1].size > 0, "dummy layer size positive");
    wmap_close(&map);
    remove(path);
}

int main(void)
{
    test_emit_layout();
    test_dummy();
    printf("llf tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
