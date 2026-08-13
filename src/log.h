#ifndef YLLM_LOG_H
#define YLLM_LOG_H

#include <stdarg.h>

enum {
    YLOG_DEBUG = 0,
    YLOG_INFO = 1,
    YLOG_WARN = 2,
    YLOG_ERROR = 3,
};

/* 打开日志文件(追加模式)。path 可含目录, 目录不存在会自动创建。
 * 之后 ylog_log 同时写入文件与控制台(stderr)。
 * 若 path 为 NULL, 仅控制台输出。默认级别 INFO。
 */
void ylog_open(const char* path);
void ylog_set_level(int level);
int  ylog_get_level(void);
/* 1 = 控制台回显(默认开), 0 = 仅写文件 */
void ylog_set_console(int on);
void ylog_log(int level, const char* fmt, ...);
void ylog_close(void);

#define ylog_debug(...) ylog_log(YLOG_DEBUG, __VA_ARGS__)
#define ylog_info(...)  ylog_log(YLOG_INFO, __VA_ARGS__)
#define ylog_warn(...)  ylog_log(YLOG_WARN, __VA_ARGS__)
#define ylog_error(...) ylog_log(YLOG_ERROR, __VA_ARGS__)

#endif