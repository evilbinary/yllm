/* 按 mmproj projector_type 分发 MiniCPM-V / Qwen3-VL / Gemma4v */
#include "vision.h"
#include "vision_impl.h"
#include "yllm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

struct Vision {
    int kind; /* 0 mcpv 1 q3v 2 g4v */
    void* p;
};

static int peek_projector(const char* path, char* out, size_t n)
{
    FILE* f;
    unsigned char hdr[24];
    uint32_t ver, typ;
    uint64_t n_tensors, n_kv, i, klen;
    if (!out || n == 0) return -1;
    out[0] = 0;
    f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(hdr, 1, 24, f) != 24 || memcmp(hdr, "GGUF", 4) != 0) {
        fclose(f); return -1;
    }
    memcpy(&ver, hdr + 4, 4);
    memcpy(&n_tensors, hdr + 8, 8);
    memcpy(&n_kv, hdr + 16, 8);
    (void)ver; (void)n_tensors;
    for (i = 0; i < n_kv; i++) {
        char key[256];
        if (fread(&klen, 8, 1, f) != 1 || klen >= sizeof(key)) { fclose(f); return -1; }
        if (fread(key, 1, (size_t)klen, f) != (size_t)klen) { fclose(f); return -1; }
        key[klen] = 0;
        if (fread(&typ, 4, 1, f) != 1) { fclose(f); return -1; }
        if (!strcmp(key, "clip.projector_type") && typ == 8) {
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1 || slen >= n) { fclose(f); return -1; }
            if (fread(out, 1, (size_t)slen, f) != (size_t)slen) { fclose(f); return -1; }
            out[slen] = 0;
            fclose(f);
            return 0;
        }
        /* skip value */
        switch (typ) {
        case 0: case 1: case 7: fseek(f, 1, SEEK_CUR); break;
        case 2: case 3: fseek(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
        case 10: case 11: case 12: fseek(f, 8, SEEK_CUR); break;
        case 8: {
            uint64_t sl; if (fread(&sl, 8, 1, f) != 1) { fclose(f); return -1; }
            fseek(f, (long)sl, SEEK_CUR); break;
        }
        case 9: {
            uint32_t at; uint64_t nn, j;
            if (fread(&at, 4, 1, f) != 1 || fread(&nn, 8, 1, f) != 1) { fclose(f); return -1; }
            for (j = 0; j < nn; j++) {
                int sz = 4;
                if (at == 0 || at == 1 || at == 7) sz = 1;
                else if (at == 2 || at == 3) sz = 2;
                else if (at == 10 || at == 11 || at == 12) sz = 8;
                else if (at == 8) {
                    uint64_t sl; if (fread(&sl, 8, 1, f) != 1) { fclose(f); return -1; }
                    fseek(f, (long)sl, SEEK_CUR); continue;
                }
                fseek(f, sz, SEEK_CUR);
            }
            break;
        }
        default: fclose(f); return -1;
        }
    }
    fclose(f);
    return 0;
}

Vision* vision_load(const char* mmproj_path, char* err, size_t errlen)
{
    Vision* v;
    char proj[64];
    if (!mmproj_path) { if (err) snprintf(err, errlen, "no mmproj"); return NULL; }
    v = (Vision*)ycalloc(1, sizeof(*v));
    if (!v) { if (err) snprintf(err, errlen, "oom"); return NULL; }
    proj[0] = 0;
    peek_projector(mmproj_path, proj, sizeof(proj));
    if (strstr(proj, "gemma4uv")) {
        if (err) snprintf(err, errlen, "gemma4uv mmproj not supported (E2B/E4B use gemma4v)");
        free(v);
        return NULL;
    }
    if (strstr(proj, "qwen3vl")) {
        v->kind = 1;
        v->p = q3v_load(mmproj_path, err, errlen);
    } else if (strstr(proj, "gemma4v")) {
        v->kind = 2;
        v->p = g4v_load(mmproj_path, err, errlen);
    } else {
        v->kind = 0;
        v->p = mcpv_load(mmproj_path, err, errlen);
    }
    if (!v->p) { free(v); return NULL; }
    return v;
}

void vision_free(Vision* v)
{
    if (!v) return;
    if (v->kind == 1) q3v_free((Q3v*)v->p);
    else if (v->kind == 2) g4v_free((G4v*)v->p);
    else mcpv_free((Mcpv*)v->p);
    free(v);
}

int vision_n_tokens(const Vision* v)
{
    if (!v || !v->p) return 0;
    if (v->kind == 1) return q3v_n_tokens((const Q3v*)v->p);
    if (v->kind == 2) return g4v_n_tokens((const G4v*)v->p);
    return mcpv_n_tokens((const Mcpv*)v->p);
}

int vision_hidden(const Vision* v)
{
    if (!v || !v->p) return 0;
    if (v->kind == 1) return q3v_hidden((const Q3v*)v->p);
    if (v->kind == 2) return g4v_hidden((const G4v*)v->p);
    return mcpv_hidden((const Mcpv*)v->p);
}

int vision_n_deepstack(const Vision* v)
{
    if (!v || !v->p || v->kind != 1) return 0;
    return q3v_n_deepstack((const Q3v*)v->p);
}

int vision_encode_image(Vision* v, const char* image_path, float* out, int max_tok,
                        char* err, size_t errlen)
{
    return vision_encode_image_ds(v, image_path, out, NULL, max_tok, err, errlen);
}

int vision_encode_image_ds(Vision* v, const char* image_path, float* out, float* ds, int max_tok,
                           char* err, size_t errlen)
{
    if (!v || !v->p) { if (err) snprintf(err, errlen, "no vision"); return -1; }
    if (v->kind == 1)
        return q3v_encode(v->p, image_path, out, ds, max_tok, err, errlen);
    (void)ds;
    if (v->kind == 2)
        return g4v_encode(v->p, image_path, out, max_tok, err, errlen);
    return mcpv_encode_image((Mcpv*)v->p, image_path, out, max_tok, err, errlen);
}
