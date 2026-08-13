#ifndef YLLM_SERVE_SERVER_H
#define YLLM_SERVE_SERVER_H

/* yllm server: 业务逻辑组(租用 rank 组, 转发请求, 广播注册/心跳) */
int cmd_server(int argc, char** argv);

#endif
