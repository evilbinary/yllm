#ifndef YLLM_SERVE_ROUTER_HTTP_H
#define YLLM_SERVE_ROUTER_HTTP_H

#include "router.h"
#include <stdint.h>

/* 启动 OpenAI 兼容 HTTP 线程(router --http-port 调用) */
int router_http_start(Router* r, uint16_t http_port);

/* API 请求/响应日志开关(1 = 开启默认, 0 = 关闭) */
void router_http_set_api_log(int on);

/* 设置 OpenAI 兼容 API key(空 = 不校验) */
void router_http_set_api_key(const char* key);

/* 设置 HTTP 入口排队上限(汇总 waiting 超过则回 429 + Retry-After; 0 = 关闭) */
void router_http_set_queue_cap(int cap);

/* 设置 RPM/TPM 滑动窗口限速(60s, 0 = 关闭) */
void router_http_set_rate_limits(int rpm, long long tpm);

#endif
