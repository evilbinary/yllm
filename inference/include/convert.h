#ifndef YLLM_CONVERT_H
#define YLLM_CONVERT_H

#include "yllm.h"
#include <stdio.h>

/* 转换器公共:一个待写入 LLF 的张量描述 */
typedef struct {
    uint32_t layer;    /* 0=embed, 1..n_blocks=transformer, n_blocks+1=final norm, n_blocks+2=output */
    uint32_t slot;     /* 层内槽位 0..8 */
    uint32_t dtype;    /* DT_* */
    uint32_t ndim;
    uint32_t shape[4];
    uint64_t nbytes;
    char name[24];
    const uint8_t* src;    /* 源数据指针 */
    uint64_t src_off;      /* 源内偏移 */
} ConvItem;

int conv_item_compare(const void* a, const void* b);
uint64_t align_up(uint64_t v, uint64_t a);

/* 按 out_dtype 重打包线性权重: DT_Q4K=保持Q4K(W4→Q4K 未实现); DT_W4B64=Q4K→W4。
 * 新分配缓冲追加到 *owned / *n_owned(调用方 free); 成功返回重打包张量数, 失败 -1。 */
int conv_items_apply_dtype(ConvItem* items, int n, uint32_t n_blocks, uint32_t out_dtype,
                           uint8_t*** owned, int* n_owned, LlfHeader* h,
                           char* err, size_t errlen);

/* 布局 + 写 header/dir/metas + 数据复制,完成后调用方无需再写文件 */
int llf_emit(const char* out_path, LlfHeader* h, ConvItem* items, int n,
             char* err, size_t errlen);

#endif
