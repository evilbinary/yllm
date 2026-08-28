#ifndef YLLM_VISION_IMPL_H
#define YLLM_VISION_IMPL_H

#include "vision.h"
#include <stddef.h>

typedef struct Mcpv Mcpv;
typedef struct Q3v Q3v;
typedef struct G4v G4v;

Mcpv* mcpv_load(const char* path, char* err, size_t errlen);
void mcpv_free(Mcpv* v);
int mcpv_n_tokens(const Mcpv* v);
int mcpv_hidden(const Mcpv* v);
int mcpv_encode_image(Mcpv* v, const char* image_path, float* out, int max_tok, char* err, size_t errlen);

Q3v* q3v_load(const char* path, char* err, size_t errlen);
void q3v_free(Q3v* v);
int q3v_n_tokens(const Q3v* v);
int q3v_hidden(const Q3v* v);
int q3v_n_deepstack(const Q3v* v);
int q3v_encode(Q3v* v, const char* image_path, float* out, float* ds, int max_tok,
               char* err, size_t errlen);

G4v* g4v_load(const char* path, char* err, size_t errlen);
void g4v_free(G4v* v);
int g4v_n_tokens(const G4v* v);
int g4v_hidden(const G4v* v);
int g4v_encode(G4v* v, const char* image_path, float* out, int max_tok, char* err, size_t errlen);
int g4v_apply_opt(G4v* v, int min_tok, int max_tok, char* err, size_t errlen);
int q3v_apply_opt(Q3v* v, int min_tok, int max_tok, char* err, size_t errlen);
int mcpv_apply_opt(Mcpv* v, int downsample, int max_slice, char* err, size_t errlen);

#endif
