#include "yllm.h"
#include "matvec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double now_ms(void)
{
    return ynow_ms();
}

/* 构造随机的 q4k block: d=1, min=0, scales=1, qs 随机 */
static void fill_q4k(uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t r, b;
    for (r = 0; r < out; r++)
        for (b = 0; b < nb; b++) {
            uint8_t* blk = w + ((size_t)r * nb + b) * 144;
            memset(blk, 0, 144);
            blk[0] = 0x00; blk[1] = 0x3c;
            blk[2] = 0x00; blk[3] = 0x00;
            uint32_t s;
            for (s = 0; s < 12; s++) blk[4 + s] = 0x01;
            for (s = 0; s < 128; s++) blk[16 + s] = (uint8_t)(rand() & 0xFF);
        }
}

static void fill_q5k(uint8_t* w, uint32_t out, uint32_t in)
{
    uint32_t nb = in / 256;
    uint32_t r, b;
    for (r = 0; r < out; r++)
        for (b = 0; b < nb; b++) {
            uint8_t* blk = w + ((size_t)r * nb + b) * 176;
            memset(blk, 0, 176);
            blk[0] = 0x00; blk[1] = 0x3c;
            blk[2] = 0x00; blk[3] = 0x00;
            uint32_t s;
            for (s = 0; s < 12; s++) blk[4 + s] = 0x01;
            for (s = 0; s < 32; s++) blk[16 + s] = (uint8_t)(rand() & 0xFF);   /* qh */
            for (s = 0; s < 128; s++) blk[48 + s] = (uint8_t)(rand() & 0xFF);  /* qs */
        }
}

static void bench_one(const char* name, uint32_t out, uint32_t in, int dtype)
{
    uint32_t rowb = (dtype == DT_Q5K) ? (in / 256) * 176 : (in / 256) * 144;
    uint8_t* w = (uint8_t*)malloc((size_t)out * rowb);
    float* x = (float*)malloc((size_t)in * 4);
    float* y = (float*)malloc((size_t)out * 4);
    if (dtype == DT_Q5K) fill_q5k(w, out, in); else fill_q4k(w, out, in);
    uint32_t i;
    for (i = 0; i < in; i++) x[i] = (float)(rand() & 0xFF) / 100.0f - 1.0f;

    /* warmup */
    matmul(y, x, w, out, in, dtype);
    int reps = 3;
    double t0 = now_ms();
    for (i = 0; i < (uint32_t)reps; i++) matmul(y, x, w, out, in, dtype);
    double ms = (now_ms() - t0) / reps;
    double gb = (double)out * rowb / 1e9;
    printf("%-8s out=%u in=%u  %.2f ms  %6.2f GB/s  y[0]=%.4f\n",
           name, out, in, ms, gb / (ms / 1000.0), y[0]);
    free(w); free(x); free(y);
}

int main(void)
{
    /* 模拟模型 FFN 维度 */
    bench_one("q4k", 5120, 5120, DT_Q4K);
    bench_one("q4k", 13696, 5120, DT_Q4K);
    bench_one("q5k", 5120, 5120, DT_Q5K);
    bench_one("q5k", 13696, 5120, DT_Q5K);
    return 0;
}