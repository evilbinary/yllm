#ifndef YLLM_SERVE_ROUTER_HTTP_H
#define YLLM_SERVE_ROUTER_HTTP_H

#include "router.h"
#include <stdint.h>

/* 启动 OpenAI 兼容 HTTP 线程(router --http-port 调用) */
int router_http_start(Router* r, uint16_t http_port);

#endif
