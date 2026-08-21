#include "log.h"
#include "yllm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
#endif

static FILE* g_fp = NULL;
static int g_level = YLOG_INFO;
static int g_console = 1;
static void* g_mu = NULL;

static void log_ensure_mu(void)
{
    if (!g_mu) ymutex_create(&g_mu);
}

static void mkdirs_for(const char* path)
{
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return;
    memcpy(buf, path, n + 1);
    size_t i;
    for (i = 0; buf[i]; i++) {
        if (buf[i] == '/' || buf[i] == '\\') {
            if (i == 0) continue;
            buf[i] = '\0';
#ifdef _WIN32
            _mkdir(buf);
#else
            mkdir(buf, 0755);
#endif
            buf[i] = '/';
        }
    }
}

void ylog_open(const char* path)
{
    log_ensure_mu();
    ymutex_lock(g_mu);
    if (g_fp) { fclose(g_fp); g_fp = NULL; }
    if (path && path[0]) {
        mkdirs_for(path);
        g_fp = fopen(path, "a");
        if (!g_fp) fprintf(stderr, "[log] cannot open log file: %s\n", path);
    }
    ymutex_unlock(g_mu);
}

void ylog_set_level(int level) { g_level = level; }
int  ylog_get_level(void)      { return g_level; }
void ylog_set_console(int on)  { g_console = on; }

/* 时间戳 YYYY-MM-DD HH:MM:SS */
static void ts(char* out, size_t n)
{
    time_t t = time(NULL);
    struct tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    snprintf(out, n, "%04d-%02d-%02d %02d:%02d:%02d",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec);
}

void ylog_log(int level, const char* fmt, ...)
{
    if (level < g_level) return;
    const char* name = "DEBUG";
    if (level == YLOG_INFO) name = "INFO";
    else if (level == YLOG_WARN) name = "WARN";
    else if (level == YLOG_ERROR) name = "ERROR";

    char stamp[32];
    ts(stamp, sizeof(stamp));

    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    log_ensure_mu();
    ymutex_lock(g_mu);
    if (g_fp) fprintf(g_fp, "[%s] [%s] %s\n", stamp, name, buf);
    if (g_console || !g_fp) {
#ifdef __ANDROID__
        int prio = 3; /* ANDROID_LOG_DEBUG */
        if (level == YLOG_INFO) prio = 4;
        else if (level == YLOG_WARN) prio = 5;
        else if (level == YLOG_ERROR) prio = 6;
        __android_log_print(prio, "yllm", "[%s] %s", name, buf);
#else
        fprintf(stderr, "[%s] [%s] %s\n", stamp, name, buf);
#endif
    }
    if (g_fp) fflush(g_fp);
    ymutex_unlock(g_mu);
}

void ylog_raw(const char* fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    log_ensure_mu();
    ymutex_lock(g_mu);
    if (g_fp) { fputs(buf, g_fp); fflush(g_fp); }
    if (g_console || !g_fp) { fputs(buf, stderr); fflush(stderr); }
    ymutex_unlock(g_mu);
}

/* 流式 token 文本, 仅写日志文件(不写 stderr)。
 * 用于 stdout 已输出 token 的进程(如 chat), 避免终端重复。 */
void ylog_raw_log(const char* fmt, ...)
{
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    log_ensure_mu();
    ymutex_lock(g_mu);
    if (g_fp) { fputs(buf, g_fp); fflush(g_fp); }
    ymutex_unlock(g_mu);
}

void ylog_close(void)
{
    if (g_mu) ymutex_lock(g_mu);
    if (g_fp) { fclose(g_fp); g_fp = NULL; }
    if (g_mu) {
        ymutex_unlock(g_mu);
        ymutex_destroy(g_mu);
        g_mu = NULL;
    }
}