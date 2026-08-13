/* protocol.h — serve 层帧协议(rank / server / router 共用)
 *
 * 协议: 一行文本命令 + 可选二进制 payload, TCP 流式(以 \n 分帧)。
 * 风格与 inference/dist.c 的私有帧一致, 但这是服务层协议:
 *   rank 对外的推理/管理帧、server↔router 的注册/心跳帧。
 *
 * 帧格式(全部以 \n 结尾的一行, payload 用长度前缀):
 *   CMD args...\n
 *   [CMD_PAYLOAD]: <n> 字节二进制紧跟在换行之后
 *
 * 错误统一回: ERR <msg>
 */

#ifndef YLLM_SERVE_PROTOCOL_H
#define YLLM_SERVE_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

/* ---- rank 帧(server / supervisor → rank) ---- */

/* PING\n → OK READY\n | OK LOADING\n | ERR <msg>
 * 存活/就绪探测(server 心跳、supervisor 自愈检测用) */
#define PROTO_PING "PING"

/* STAT\n → OK inflight=<n> kv_mb=<f> prefix_hits=<n> uptime_s=<t>\n
 * 状态上报(供 server 路由决策: 忙闲 / KV 占用) */
#define PROTO_STAT "STAT"

/* INFER <max_tokens> <n_bytes>\n<prompt bytes>
 * 生成请求: max_tokens 上限, n_bytes = prompt 的 UTF-8 字节数。
 * 响应: 逐 token 流式回 T <len>\n<token utf8 bytes>...,
 *       结束回 DONE <gen_tokens> <eos=0|1> <ms>\n
 * 错误回 ERR <msg> */
#define PROTO_INFER "INFER"

/* DRAIN\n → OK\n(等当前请求完成) 后关连接退出
 * 优雅下线(滚动更新 / 缩容) */
#define PROTO_DRAIN "DRAIN"

/* QUIT\n → OK\n 后退出(仅测试) */
#define PROTO_QUIT "QUIT"

/* ---- server 帧(router / supervisor → server) ---- */

/* REGISTER server-<id> model=<name> leader=<ip:port> ranks=<n>\n → OK\n
 * server 启动时广播到所有 router(注册业务组) */
#define PROTO_REGISTER "REGISTER"

/* HEARTBEAT <server-id> inflight=<n> kv_mb=<f>\n → OK\n
 * 周期性广播(每 2s), router 据此维护注册表状态 */
#define PROTO_HEARTBEAT "HEARTBEAT"

/* SCALE <server-id> need_groups=<n>\n → OK\n
 * server 向 supervisor 请求扩/缩容(增加/减少 rank 组) */
#define PROTO_SCALE "SCALE"

/* ---- 流式 token 帧(server → router 透传, rank → server) ---- */

/* T <len>\n<token utf8 bytes>: 生成的一个 token */
#define PROTO_TOKEN "T"

/* DONE <gen_tokens> <eos=0|1> <ms>\n: 生成结束 */
#define PROTO_DONE "DONE"

/* ---- 通用解析辅助 ---- */

#include <string.h>

/* 解析 "key=value" 形式的参数: 返回 value 指针或 NULL */
static inline const char* proto_get(const char* line, const char* key)
{
    size_t klen = strlen(key);
    const char* p = line;
    while (*p) {
        while (*p == ' ') p++;
        if (strncmp(p, key, klen) == 0 && p[klen] == '=') {
            const char* v = p + klen + 1;
            return v;
        }
        while (*p && *p != ' ') p++;
    }
    return NULL;
}

#endif /* YLLM_SERVE_PROTOCOL_H */
