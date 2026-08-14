/* frame.h — 统一帧协议(rank / server / router / supervisor 所有节点间通信)
 *
 * 帧格式: 一行文本 "CMD v=<ver> args\n"(args 可含 key=value),
 *         可选二进制 payload(长度前缀, 紧跟换行之后)。
 *
 * 例子:
 *   HEARTBEAT v=1 rank-r0 type=rank state=ready inflight=0 kv_mb=601.0
 *   INFER v=1 20 15\n<15 bytes prompt>
 *   SERVER_ADD v=1 server-a model=tinyllama leader=127.0.0.1:9420
 *
 * 版本策略(帧级): 每个 frame 都带 v=<PROTO_VERSION>, 接收方解析即校验。
 * 不匹配(含旧客户端无 v=)拒绝该帧并记日志, 避免新旧混跑解析错乱。
 *
 * 依赖: sock.h(传输层), protocol.h(版本号)
 */

#ifndef YLLM_SERVE_FRAME_H
#define YLLM_SERVE_FRAME_H

#include "sock.h"
#include "protocol.h"
#include "../inference/log.h"
#include <stddef.h>

#define FRAME_CMD_MAX 256
#define FRAME_ARGS_MAX 2048

typedef struct {
    char cmd[FRAME_CMD_MAX];
    char args[FRAME_ARGS_MAX];
} Frame;

/* 发送一帧(纯命令行): 自动附加 v=<版本> 在 args 开头 */
static inline int frame_send(int fd, const char* cmd, const char* args)
{
    sock_send_n(fd, cmd, strlen(cmd));
    sock_send_n(fd, " v=", 3);
    sock_send_n(fd, PROTO_VERSION_STR, strlen(PROTO_VERSION_STR));
    if (args && args[0]) {
        sock_send_n(fd, " ", 1);
        sock_send_n(fd, args, strlen(args));
    }
    sock_send_n(fd, "\n", 1);
    return 0;
}

/* 接收一帧(阻塞读一行, 拆成 cmd + args), 并校验 v= 版本。
 * 版本不匹配或缺失: 记日志并返回 -1(调用方应断开)。
 * 注意: 校验失败会消耗掉该行, 连接不可继续使用。 */
static inline int frame_recv(int fd, Frame* f)
{
    char line[FRAME_CMD_MAX + FRAME_ARGS_MAX + 2];
    int n = sock_recv_line(fd, line, sizeof(line));
    if (n < 0) return -1;
    const char* sp = strchr(line, ' ');
    if (sp) {
        size_t clen = (size_t)(sp - line);
        if (clen >= FRAME_CMD_MAX) clen = FRAME_CMD_MAX - 1;
        memcpy(f->cmd, line, clen);
        f->cmd[clen] = '\0';
        snprintf(f->args, sizeof(f->args), "%s", sp + 1);
    } else {
        snprintf(f->cmd, sizeof(f->cmd), "%s", line);
        f->args[0] = '\0';
    }
    /* 校验 v= 版本(必须在 args 首字段) */
    char ver[64] = "";
    if (sscanf(f->args, "v=%63s", ver) != 1) {
        ylog_warn("[proto] %s: missing v= version field (frame=[%s])", f->cmd, line);
        return -1;
    }
    if (strcmp(ver, PROTO_VERSION_STR) != 0) {
        ylog_warn("[proto] %s: version mismatch (peer=%s, me=%s)", f->cmd, ver, PROTO_VERSION_STR);
        return -1;
    }
    /* 剥离 v= 前缀, 使 args 与旧调用方解析兼容 */
    const char* rest = strchr(f->args, ' ');
    if (rest) {
        while (*rest == ' ') rest++;
        char tmp[FRAME_ARGS_MAX];
        snprintf(tmp, sizeof(tmp), "%s", rest);
        snprintf(f->args, sizeof(f->args), "%s", tmp);
    } else {
        f->args[0] = '\0';
    }
    return 0;
}

/* 发送带 payload 的帧: 先发 "CMD v=<ver> <n_bytes>\n", 再发 n 字节 payload */
static inline int frame_send_payload(int fd, const char* cmd, const char* args,
                                     const void* payload, size_t n)
{
    char hdr[FRAME_CMD_MAX + 64];
    int hn;
    if (args && args[0])
        hn = snprintf(hdr, sizeof(hdr), "%s v=%s %s %zu\n", cmd, PROTO_VERSION_STR, args, n);
    else
        hn = snprintf(hdr, sizeof(hdr), "%s v=%s %zu\n", cmd, PROTO_VERSION_STR, n);
    sock_send_n(fd, hdr, (size_t)hn);
    if (n > 0) sock_send_n(fd, payload, n);
    return 0;
}

/* 接收带 payload 的帧: 调用方先 frame_recv 拿到行(含长度), 再调本函数读 payload。
 * buf 需至少可容纳 n 字节。 */
static inline int frame_recv_payload(int fd, void* buf, size_t n)
{
    if (n > 0 && sock_recv_n(fd, buf, n) != 0) return -1;
    return 0;
}

/* 读取 "T <len>" 帧的 payload: line 为刚收到的帧头行。
 * 若 payload 超过 line 缓冲则用堆(调用方必须 free), 否则直接写入 line。
 * 成功返回 payload 指针并置 *plen, 失败返回 NULL。 */
static inline char* frame_t_payload(int fd, char* line, size_t linesz, size_t* plen)
{
    long tlen = atol(line + 2);
    if (tlen <= 0) return NULL;
    *plen = (size_t)tlen;
    if ((size_t)tlen < linesz) {
        if (sock_recv_n(fd, line, (size_t)tlen) != 0) return NULL;
        line[tlen] = '\0';
        return line;
    }
    char* p = (char*)malloc((size_t)tlen + 1);
    if (!p) return NULL;
    if (sock_recv_n(fd, p, (size_t)tlen) != 0) { free(p); return NULL; }
    p[tlen] = '\0';
    return p;
}

/* 解析 "key=value" 参数: 复制 value 到 out(截断到空格/结尾), 返回 0 或 -1 */
static inline int frame_get(const Frame* f, const char* key, char* out, size_t outsz)
{
    size_t klen = strlen(key);
    const char* p = f->args;
    while (*p) {
        while (*p == ' ') p++;
        if (strncmp(p, key, klen) == 0 && p[klen] == '=') {
            const char* v = p + klen + 1;
            size_t vlen = 0;
            while (v[vlen] && v[vlen] != ' ') vlen++;
            if (vlen >= outsz) vlen = outsz - 1;
            memcpy(out, v, vlen);
            out[vlen] = '\0';
            return 0;
        }
        while (*p && *p != ' ') p++;
    }
    out[0] = '\0';
    return -1;
}

/* 取 payload 长度(args 末尾的裸数字, 如 "INFER v=1 20 15" 的 15) */
static inline long frame_payload_len(const Frame* f)
{
    const char* p = f->args;
    const char* last = NULL;
    while (*p) {
        while (*p == ' ') p++;
        const char* tok = p;
        while (*p && *p != ' ') p++;
        if (p > tok) last = tok;
    }
    return last ? atol(last) : 0;
}

#endif /* YLLM_SERVE_FRAME_H */
