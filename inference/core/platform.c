#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "yllm.h"
#include "log.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

void* ymalloc(size_t n)
{
    void* p = malloc(n);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

void* ycalloc(size_t n, size_t s)
{
    void* p = calloc(n, s);
    if (!p) { fprintf(stderr, "out of memory\n"); exit(1); }
    return p;
}

char* ystrdup(const char* s)
{
    size_t n = strlen(s) + 1;
    char* d = ymalloc(n);
    memcpy(d, s, n);
    return d;
}

uint64_t ynow_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000 / f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
#endif
}

uint64_t ynow_ns(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)(c.QuadPart * 1000000000ull / f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
#endif
}

void ymsleep(uint32_t ms)
{
#ifdef _WIN32
    Sleep(ms);
#else
    usleep((useconds_t)ms * 1000);
#endif
}

void yllm_tune_cpu(void)
{
    static int once;
    if (once) return;
    once = 1;
    if (getenv("YLLM_NO_AFFINITY")) return;
#if defined(__linux__)
    {
        unsigned long khz[64];
        int n = 0, i;
        unsigned long minf = 0, maxf = 0;
        for (i = 0; i < 64; i++) {
            char path[96];
            FILE* f;
            unsigned long hz = 0;
            snprintf(path, sizeof(path),
                     "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
            f = fopen(path, "r");
            if (!f) break;
            if (fscanf(f, "%lu", &hz) != 1) hz = 0;
            fclose(f);
            khz[n++] = hz;
        }
        if (n <= 1) return;
        minf = maxf = khz[0];
        for (i = 1; i < n; i++) {
            if (khz[i] < minf) minf = khz[i];
            if (khz[i] > maxf) maxf = khz[i];
        }
        if (maxf > minf) {
            unsigned long thresh = minf + (maxf - minf) / 2;
            cpu_set_t set;
            int nbig = 0;
            char buf[128];
            int off = 0;
            CPU_ZERO(&set);
            buf[0] = 0;
            for (i = 0; i < n; i++) {
                if (khz[i] >= thresh) {
                    CPU_SET(i, &set);
                    nbig++;
                    if (off < (int)sizeof(buf) - 8)
                        off += snprintf(buf + off, sizeof(buf) - (size_t)off, "%s%d", off ? "," : "", i);
                }
            }
            if (nbig > 0 && nbig < n && sched_setaffinity(0, sizeof(set), &set) == 0) {
#ifdef _OPENMP
                if (!getenv("OMP_NUM_THREADS"))
                    omp_set_num_threads(nbig);
#endif
                ylog_info("cpu: affinity [%s] (%d/%d cores, %lu-%lu kHz)",
                          buf, nbig, n, minf, maxf);
            }
        }
    }
#endif
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
}

#ifdef _WIN32
typedef struct { void (*fn)(void*); void* arg; } YTArg;
static DWORD WINAPI yt_main(LPVOID p)
{
    YTArg* a = (YTArg*)p;
    a->fn(a->arg);
    free(a);
    return 0;
}
int ythread_create(void* t, void (*fn)(void*), void* arg)
{
    YTArg* a = ymalloc(sizeof(YTArg));
    a->fn = fn;
    a->arg = arg;
    HANDLE h = CreateThread(NULL, 0, yt_main, a, 0, NULL);
    if (!h) { free(a); return -1; }
    *(HANDLE*)t = h;
    return 0;
}
void ythread_join(void* t)
{
    HANDLE h = *(HANDLE*)t;
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
}
#else
typedef struct { void (*fn)(void*); void* arg; } YTArg;
static void* yt_main(void* p)
{
    YTArg* a = (YTArg*)p;
    a->fn(a->arg);
    free(a);
    return NULL;
}
int ythread_create(void* t, void (*fn)(void*), void* arg)
{
    YTArg* a = ymalloc(sizeof(YTArg));
    a->fn = fn;
    a->arg = arg;
    if (pthread_create((pthread_t*)t, NULL, yt_main, a) != 0) { free(a); return -1; }
    return 0;
}
void ythread_join(void* t)
{
    pthread_join(*(pthread_t*)t, NULL);
}
#endif

void ymutex_create(void** m)
{
#ifdef _WIN32
    CRITICAL_SECTION* c = (CRITICAL_SECTION*)ymalloc(sizeof(CRITICAL_SECTION));
    InitializeCriticalSection(c);
    *m = c;
#else
    pthread_mutex_t* c = (pthread_mutex_t*)ymalloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(c, NULL);
    *m = c;
#endif
}

void ymutex_lock(void* m)
{
#ifdef _WIN32
    EnterCriticalSection((CRITICAL_SECTION*)m);
#else
    pthread_mutex_lock((pthread_mutex_t*)m);
#endif
}

void ymutex_unlock(void* m)
{
#ifdef _WIN32
    LeaveCriticalSection((CRITICAL_SECTION*)m);
#else
    pthread_mutex_unlock((pthread_mutex_t*)m);
#endif
}

void ymutex_destroy(void* m)
{
#ifdef _WIN32
    DeleteCriticalSection((CRITICAL_SECTION*)m);
#else
    pthread_mutex_destroy((pthread_mutex_t*)m);
#endif
    free(m);
}

int yfile_size(const char* path, uint64_t* size)
{
#ifdef _WIN32
    wchar_t wp[2048];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 2048);
    HANDLE hf = CreateFileW(wp, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hf == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz)) { CloseHandle(hf); return -1; }
    *size = (uint64_t)sz.QuadPart;
    CloseHandle(hf);
    return 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    *size = (uint64_t)st.st_size;
    return 0;
#endif
}

#ifdef _WIN32
typedef struct {
    PVOID VirtualAddress;
    SIZE_T NumberOfBytes;
} YWinMemRange;

static int win_prefetch(void* addr, size_t n)
{
    typedef BOOL (WINAPI *PrefetchFn)(HANDLE, ULONG_PTR, YWinMemRange*, ULONG);
    static PrefetchFn fn;
    static int inited;
    const size_t chunk = (size_t)128 * 1024 * 1024;
    if (!inited) {
        HMODULE k = GetModuleHandleW(L"kernel32.dll");
        if (k) fn = (PrefetchFn)GetProcAddress(k, "PrefetchVirtualMemory");
        inited = 1;
    }
    if (!fn || !addr || n == 0) return -1;
    uint8_t* p = (uint8_t*)addr;
    while (n > 0) {
        YWinMemRange r;
        size_t m = n < chunk ? n : chunk;
        r.VirtualAddress = p;
        r.NumberOfBytes = m;
        if (!fn(GetCurrentProcess(), 1, &r, 0)) return -1;
        p += m;
        n -= m;
    }
    return 0;
}
#endif

int wmap_open(const char* path, WMap* m)
{
    memset(m, 0, sizeof(WMap));
    if (yfile_size(path, &m->size) != 0) return -1;
#ifdef _WIN32
    wchar_t wp[2048];
    MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 2048);
    m->hfile = CreateFileW(wp, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (m->hfile == INVALID_HANDLE_VALUE) return -1;
    m->hmap = CreateFileMappingW((HANDLE)m->hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->hmap) { CloseHandle((HANDLE)m->hfile); m->hfile = NULL; return -1; }
    m->base = MapViewOfFile((HANDLE)m->hmap, FILE_MAP_READ, 0, 0, 0);
    if (!m->base) { CloseHandle((HANDLE)m->hmap); CloseHandle((HANDLE)m->hfile); m->hmap = NULL; m->hfile = NULL; return -1; }
    win_prefetch(m->base, (size_t)m->size);
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    m->base = mmap(NULL, m->size, PROT_READ, MAP_SHARED, m->fd, 0);
    if (m->base == MAP_FAILED) { close(m->fd); m->fd = -1; return -1; }
#ifdef __linux__
    madvise(m->base, (size_t)m->size, MADV_WILLNEED);
#endif
#endif
    return 0;
}

void wmap_close(WMap* m)
{
#ifdef _WIN32
    if (m->base) UnmapViewOfFile(m->base);
    if (m->hmap) CloseHandle((HANDLE)m->hmap);
    if (m->hfile) CloseHandle((HANDLE)m->hfile);
#else
    if (m->base) munmap(m->base, m->size);
    if (m->fd >= 0) close(m->fd);
#endif
    memset(m, 0, sizeof(WMap));
}

void ws_prefetch(const Ws* ws, uint32_t layer)
{
    uint64_t off = ws->model.dir[layer].offset;
    uint64_t sz = ws->model.dir[layer].size;
    if (sz == 0 || !ws->map.base) return;
#ifdef __linux__
    madvise((char*)ws->map.base + off, (size_t)sz, MADV_WILLNEED);
#elif defined(_WIN32)
    win_prefetch((char*)ws->map.base + off, (size_t)sz);
#else
    (void)off;
#endif
}

void ws_release(const Ws* ws, uint32_t layer)
{
#ifdef __linux__
    uint64_t off = ws->model.dir[layer].offset;
    uint64_t sz = ws->model.dir[layer].size;
    if (sz == 0) return;
    /* 丢 PTE、保页缓存: munmap + MAP_FIXED 重映射。
     * 页缓存仍在 → 下次缺页是 minor fault(内存带宽级), 不会触发磁盘读。
     * 层偏移/大小在 llf 写入时保证 4096 对齐。 */
    uint8_t* p = (uint8_t*)ws->map.base + off;
    if (munmap(p, (size_t)sz) == 0) {
        void* r = mmap(p, (size_t)sz, PROT_READ, MAP_SHARED | MAP_FIXED,
                       ws->map.fd, (off_t)off);
        if (r == MAP_FAILED)
            fprintf(stderr, "ws_release remap failed layer=%u\n", layer);
    }
#else
    (void)ws;
    (void)layer;
#endif
}

#if YLLM_TENSOR_STREAM
void ws_prefetch_range(const Ws* ws, uint64_t off, uint64_t sz)
{
    if (sz == 0 || !ws->map.base) return;
#ifdef __linux__
    madvise((char*)ws->map.base + off, (size_t)sz, MADV_WILLNEED);
#elif defined(_WIN32)
    win_prefetch((char*)ws->map.base + off, (size_t)sz);
#else
    (void)off;
#endif
}

/* 页对齐范围的 munmap+mmap 释放(要求 off 与 sz 均 4096 对齐)。
 * 用于 lm_head 分块: 分块边界取整到页, 尾块保留驻留。 */
void ws_release_aligned(const Ws* ws, uint64_t off, uint64_t sz)
{
#ifdef __linux__
    if (sz == 0) return;
    uint8_t* p = (uint8_t*)ws->map.base + off;
    if (munmap(p, (size_t)sz) == 0) {
        void* r = mmap(p, (size_t)sz, PROT_READ, MAP_SHARED | MAP_FIXED,
                       ws->map.fd, (off_t)off);
        if (r == MAP_FAILED)
            fprintf(stderr, "ws_release_aligned remap failed off=%llu\n",
                    (unsigned long long)off);
    }
#else
    (void)ws; (void)off; (void)sz;
#endif
}
#endif

const void* ws_layer_ptr(const Ws* ws, uint32_t layer)
{
    return (const uint8_t*)ws->map.base + ws->model.dir[layer].offset;
}

/* 查询 [off, off+sz) 是否完全驻留在页缓存: 1=完全驻留, 0=未完全, -1=平台不支持 */
int wmap_resident(const WMap* m, uint64_t off, uint64_t sz)
{
#ifdef __linux__
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;
    size_t nvec = (size_t)((sz + (uint64_t)pg - 1) / (uint64_t)pg);
    if (nvec == 0) return 1;
    unsigned char* vec = (unsigned char*)ymalloc(nvec);
    if (mincore((char*)m->base + off, (size_t)sz, vec) != 0) { free(vec); return -1; }
    int all = 1;
    size_t i;
    for (i = 0; i < nvec; i++) {
        if (!(vec[i] & 1)) { all = 0; break; }
    }
    free(vec);
    return all;
#else
    (void)m; (void)off; (void)sz;
    return -1;
#endif
}

float f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000) << 16;
    uint32_t e = h & 0x7c00;
    uint32_t m = h & 0x3ff;
    uint32_t x;
    if (e == 0) {
        if (m == 0) x = 0;
        else {
            uint32_t mm = m;
            int sh = 0;
            while (!(mm & 0x400)) { mm <<= 1; sh++; }
            x = s | ((uint32_t)(113 - sh) << 23) | ((mm & 0x3ff) << 13);
        }
    } else if (e == 0x7c00) {
        x = s | 0x7f800000 | (m ? 0x7fc000u : 0);
    } else {
        x = s | ((e + 0x1c000u) << 13) | (m << 13);
    }
    float f;
    memcpy(&f, &x, 4);
    return f;
}

uint16_t f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    uint16_t h;
    if (((x >> 23) & 0xff) == 0) {
        h = (uint16_t)sign; /* zero or f32 subnormal -> fp16 zero */
    } else if (((x >> 23) & 0xff) == 0xff) {
        h = (uint16_t)(sign | 0x7c00 | (mant ? 0x0200 : 0));
    } else if (exp >= 31) {
        h = (uint16_t)(sign | 0x7c00);
    } else if (exp <= 0) {
        if (exp < -10) {
            h = (uint16_t)sign;
        } else {
            mant |= 0x800000;
            uint32_t shift = (uint32_t)(14 - exp);
            uint32_t round_bit = 1U << (shift - 1);
            mant = (mant + round_bit) >> shift;
            h = (uint16_t)(sign | mant);
        }
    } else {
        mant += 0x00001000;
        if (mant & 0x00800000) {
            mant = 0;
            exp++;
            if (exp >= 31) {
                h = (uint16_t)(sign | 0x7c00);
            } else {
                h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
            }
        } else {
            h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        }
    }
    return h;
}

uint16_t bf16_to_f16(uint16_t b)
{
    int e = (b >> 7) & 0xff;
    uint16_t s = b & 0x8000;
    uint16_t m = b & 0x7f;
    int ne = e - 112;
    if (e == 0xff) return (uint16_t)(s | (m ? 0x7e00 : 0x7c00));
    if (ne <= 0) return s;
    if (ne >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | ((uint16_t)ne << 10) | (uint16_t)(m << 3));
}

void f32_to_f16_buf(const float* src, uint16_t* dst, size_t n)
{
    size_t i = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    for (; i + 8 <= n; i += 8) {
        float16x4_t h0 = vcvt_f16_f32(vld1q_f32(src + i));
        float16x4_t h1 = vcvt_f16_f32(vld1q_f32(src + i + 4));
        vst1q_f16((__fp16*)(dst + i), vcombine_f16(h0, h1));
    }
#endif
    for (; i < n; i++) dst[i] = f32_to_f16(src[i]);
}

void bf16_to_f16_buf(const uint16_t* src, uint16_t* dst, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) dst[i] = bf16_to_f16(src[i]);
}

uint64_t ysrand(uint64_t seed)
{
    if (seed == 0) seed = 0x9e3779b97f4a7c15ull;
    return seed;
}

uint64_t yrng(uint64_t* s)
{
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

uint64_t yproc_rss(void)
{
#ifdef __linux__
    long v = 0;
    FILE* f = fopen("/proc/self/status", "r");
    char l[256];
    if (f) {
        while (fgets(l, sizeof l, f))
            if (!strncmp(l, "VmRSS:", 6)) { sscanf(l + 6, "%ld", &v); break; }
        fclose(f);
    }
    if (v > 0) return (uint64_t)v * 1024;
#endif
#ifdef _WIN32
    {
        typedef struct {
            DWORD cb;
            DWORD PageFaultCount;
            SIZE_T PeakWorkingSetSize;
            SIZE_T WorkingSetSize;
            SIZE_T QuotaPeakPagedPoolUsage;
            SIZE_T QuotaPagedPoolUsage;
            SIZE_T QuotaPeakNonPagedPoolUsage;
            SIZE_T QuotaNonPagedPoolUsage;
            SIZE_T PagefileUsage;
            SIZE_T PeakPagefileUsage;
        } YPmc;
        typedef BOOL (WINAPI *GpmiFn)(HANDLE, YPmc*, DWORD);
        static GpmiFn gpmi;
        static int inited;
        YPmc pmc;
        if (!inited) {
            HMODULE p = LoadLibraryW(L"psapi.dll");
            if (p) gpmi = (GpmiFn)GetProcAddress(p, "GetProcessMemoryInfo");
            inited = 1;
        }
        memset(&pmc, 0, sizeof(pmc));
        pmc.cb = sizeof(pmc);
        if (gpmi && gpmi(GetCurrentProcess(), &pmc, sizeof(pmc)))
            return (uint64_t)pmc.WorkingSetSize;
    }
#endif
    return 0;
}
