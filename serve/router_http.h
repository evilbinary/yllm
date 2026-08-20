#ifndef YLLM_SERVE_ROUTER_HTTP_H
#define YLLM_SERVE_ROUTER_HTTP_H

#include "router.h"
#include <stdint.h>

/* 启动 OpenAI 兼容 HTTP 线程(router --http-port 调用) */
int router_http_start(Router* r, uint16_t http_port);

/* API 请求/响应日志开关(1 = 开启默认, 0 = 关闭) */
void router_http_set_api_log(int on);

#endif
