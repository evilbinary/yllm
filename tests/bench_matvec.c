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

/* 64MB 刷洗缓冲: 每次 rep 前触碰, 赶出 L3 残留, 测真实 DRAM 带宽 */
static char flushbuf[64 * 1024 * 1024];

static int cmp_ms(const void* a, const void* b)
{
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
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

    matmul(y, x, w, out, in, dtype);   /* warmup */
    uint32_t j;
    for (j = 0; j < sizeof(flushbuf); j += 4096) flushbuf[j] = (char)j;  /* 驻留 */

    int reps = 7;
    double times[16];
    for (i = 0; i < (uint32_t)reps; i++) {
        for (j = 0; j < sizeof(flushbuf); j += 64) flushbuf[j] ^= 1;     /* 刷洗 L3 */
        double s = now_ms();
        matmul(y, x, w, out, in, dtype);
        times[i] = now_ms() - s;
    }
    qsort(times, (size_t)reps, sizeof(double), cmp_ms);
    double med = times[reps / 2];
    double gb = (double)out * rowb / 1e9;
    printf("%-8s out=%u in=%u  %6.2f ms(med)  %6.2f GB/s  y[0]=%.4f\n",
           name, out, in, med, gb / (med / 1000.0), y[0]);
    free(w); free(x); free(y);
}

/* 批量路径对比: int8 激活内核(matmul_batch → q4k_i8) vs f32 反量化基线。
 * 形状取 tinyllama-1.1B 实际层(Q/O: 2048→2048, gate/up: 2048→5632, down: 5632→2048)。 */
static void bench_batch(const char* name, uint32_t out, uint32_t in, uint32_t B)
{
    uint32_t rowb = (size_t)(in / 256) * 144;
    uint8_t* w = (uint8_t*)malloc((size_t)out * rowb);
    float* x = (float*)malloc((size_t)B * in * 4);
    float* y = (float*)malloc((size_t)B * out * 4);
    float* y2 = (float*)malloc((size_t)B * out * 4);
    uint32_t i, j;
    fill_q4k(w, out, in);
    for (i = 0; i < B * in; i++) x[i] = (float)(rand() & 0xFF) / 100.0f - 1.0f;

    matmul_batch(y, x, w, out, in, DT_Q4K, B);       /* warmup(int8 内核) */
    matmul_batch_q(y2, x, w, out, in, DT_Q4K, B);    /* warmup(f32 基线) */

    int reps = 7;
    const int iters = 20;    /* 每次测量内循环多次, 消除 ~1ms 计时粒度 */
    double t8[16], tq[16];
    for (i = 0; i < (uint32_t)reps; i++) {
        uint32_t k;
        for (j = 0; j < sizeof(flushbuf); j += 64) flushbuf[j] ^= 1;
        double s = now_ms();
        for (k = 0; k < iters; k++) matmul_batch(y, x, w, out, in, DT_Q4K, B);
        t8[i] = (now_ms() - s) / iters;
        for (j = 0; j < sizeof(flushbuf); j += 64) flushbuf[j] ^= 1;
        s = now_ms();
        for (k = 0; k < iters; k++) matmul_batch_q(y2, x, w, out, in, DT_Q4K, B);
        tq[i] = (now_ms() - s) / iters;
    }
    qsort(t8, reps, sizeof(double), cmp_ms);
    qsort(tq, reps, sizeof(double), cmp_ms);
    double m8 = t8[reps / 2], mq = tq[reps / 2];
    /* 一致性抽查: 两路径同批输出(激活量化路径相同, 允许浮点结合差) */
    double maxd = 0.0;
    for (i = 0; i < B * out; i++) {
        double d = fabs(y[i] - y2[i]);
        if (d > maxd) maxd = d;
    }
    printf("%-10s out=%-6u in=%-5u B=%-2u  int8 %7.2f ms  f32 %7.2f ms  speedup %.2fx  max|dy|=%.4f\n",
           name, out, in, B, m8, mq, mq / m8, maxd);
    free(w); free(x); free(y); free(y2);
}

int main(void)
{
    /* 模拟模型 FFN 维度 */
    bench_one("q4k", 5120, 5120, DT_Q4K);
    bench_one("q4k", 13696, 5120, DT_Q4K);
    bench_one("q4k", 25600, 5120, DT_Q4K);
    bench_one("q5k", 5120, 5120, DT_Q5K);
    bench_one("q5k", 13696, 5120, DT_Q5K);
    printf("\n--- batch int8 vs f32 (Q4K, tinyllama layer shapes) ---\n");
    bench_batch("q/o", 2048, 2048, 1);
    bench_batch("q/o", 2048, 2048, 4);
    bench_batch("q/o", 2048, 2048, 8);
    bench_batch("gate/up", 5632, 2048, 1);
    bench_batch("gate/up", 5632, 2048, 4);
    bench_batch("gate/up", 5632, 2048, 8);
    bench_batch("down", 2048, 5632, 1);
    bench_batch("down", 2048, 5632, 4);
    bench_batch("down", 2048, 5632, 8);
    return 0;
}