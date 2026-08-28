#ifndef YLLM_VISION_H
#define YLLM_VISION_H

#include <stddef.h>
#include <stdint.h>

typedef struct Vision Vision;

Vision* vision_load(const char* mmproj_path, char* err, size_t errlen);
void vision_free(Vision* v);
int vision_n_tokens(const Vision* v);
int vision_hidden(const Vision* v);
int vision_n_deepstack(const Vision* v);
/* 单图编码. out 长度 ≥ n_tokens*hidden。
 * ds 非空且 n_deepstack>0 时写入 [n_deepstack][n_tokens][hidden]。 */
int vision_encode_image(Vision* v, const char* image_path, float* out, int max_tok,
                        char* err, size_t errlen);
int vision_encode_image_ds(Vision* v, const char* image_path, float* out, float* ds, int max_tok,
                           char* err, size_t errlen);

#endif
