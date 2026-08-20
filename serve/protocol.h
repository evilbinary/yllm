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

/* 会话数据包调试: 编译时 -DYLLM_SESS_DEBUG=1 开启,
 * 打印 router/server/rank 各跳的实际数据包内容。 */
#ifndef YLLM_SESS_DEBUG
#define YLLM_SESS_DEBUG 0
#endif

#include <stdint.h>
#include <stddef.h>

/* ---- 协议版本(节点间通信握手用) ----
 * 版本不匹配的节点拒绝通信, 避免新旧混跑解析错乱。
 * 改动帧格式/语义时递增此号。 */
#define PROTO_VERSION 1
#define PROTO_VERSION_STR "yllm-1"

/* HELLO <version> <node_id> <type>\n → HELLO_OK <version>\n | ERR version mismatch\n
 * 连接建立后的第一条帧, 双方交换版本与身份。 */
#define PROTO_HELLO "HELLO"
#define PROTO_HELLO_OK "HELLO_OK"

/* ---- rank 帧(server / supervisor → rank) ---- */

/* PING\n → OK READY\n | OK LOADING\n | ERR <msg>
 * 存活/就绪探测(server 心跳、supervisor 自愈检测用) */
#define PROTO_PING "PING"

/* STAT\n → OK inflight=<n> kv_mb=<f> prefix_hits=<n> uptime_s=<t>\n
 * 状态上报(供 server 路由决策: 忙闲 / KV 占用) */
#define PROTO_STAT "STAT"

/* INFER <max_tokens> <n_bytes> seg=<r> segs=<n> peers=<ip1,ip2,...>\n<prompt bytes>
 * 生成请求: max_tokens 上限, n_bytes = prompt 的 UTF-8 字节数。
 * seg/segs/peers 为组内 rank 信息(由 server 从 supervisor 租用时获得, 随请求捎给 rank0):
 *   seg 段号(rank0=0), segs 总段数, peers 各段节点 IP(逗号分隔, 段号顺序)。
 * rank0 收到后按该数据组织组内协作; worker 段不接收 INFER。
 * 响应: 逐 token 流式回 T <len>\n<token utf8 bytes>...,
 *       结束回 DONE <gen_tokens> <eos=0|1> <ms>\n
 * 错误回 ERR <msg> */
#define PROTO_INFER "INFER"
#define PROTO_INFER_SESS "INFER_SESS"

/* DRAIN\n → OK\n(等当前请求完成) 后关连接退出
 * 优雅下线(滚动更新 / 缩容) */
#define PROTO_DRAIN "DRAIN"

/* QUIT\n → OK\n 后退出(仅测试) */
#define PROTO_QUIT "QUIT"

/* ---- server 帧(router / supervisor → server) ---- */

/* REGISTER server-<id> model=<name> leader=<ip:port> ranks=<n>\n → OK\n
 * server 启动时广播到所有 router(注册业务组) */
#define PROTO_REGISTER "REGISTER"

/* ---- 统一心跳(所有常驻进程共用, 数据面: 活着就报) ----
 *
 * HEARTBEAT <node-id> type=<rank|server|router|supervisor> state=<loading|ready|busy|dead>
 *            inflight=<n> kv_mb=<f> model=<name>\n → OK\n
 *
 * 发送方(数据面): 每个进程周期上报(默认 2s)。
 * 接收方(监控面): supervisor 收全部; router 收 type=server; server 收 type=rank。
 * 判死: 接收方超时未收 → 标记 DEAD → 通知 supervisor 重拉(生命周期面, 永远归 supervisor)。
 */
#define PROTO_HEARTBEAT "HEARTBEAT"

/* SCALE <server-id> need_groups=<n>\n → OK\n
 * server 向 supervisor 请求扩/缩容(增加/减少 rank 组) */
#define PROTO_SCALE "SCALE"

/* ---- server 帧(supervisor → router, 生命周期面: 注册表增删/状态推送) ---- */

/* SERVER_ADD <server-id> model=<name> leader=<ip:port>\n → OK\n
 * supervisor 通知 router: 新 server 就绪, 加入候选表 */
#define PROTO_SERVER_ADD "SERVER_ADD"

/* SERVER_DEL <server-id>\n → OK\n
 * supervisor 通知 router: server 已死/下线, 剔除 */
#define PROTO_SERVER_DEL "SERVER_DEL"

/* SERVER_UPDATE <server-id> inflight=<n> kv_mb=<f>\n → OK\n
 * supervisor 推送 server 运行状态(路由决策用, 方式A) */
#define PROTO_SERVER_UPDATE "SERVER_UPDATE"

/* QUERY_SERVERS\n → SERVER_INFO ... QUERY_DONE\n
 * router 主动查询 supervisor 的 server 快照(方式B) */
#define PROTO_QUERY_SERVERS "QUERY_SERVERS"
#define PROTO_SERVER_INFO "SERVER_INFO"
#define PROTO_QUERY_DONE "QUERY_DONE"

/* QUERY_RANKS model=<name>\n → RANK_INFO ... QUERY_DONE\n
 * server 主动查询 supervisor: 该模型所有 ready rank 的真实地址(自动发现,
 * 跨机器分布式: rank 心跳上报自身 IP, server 无需配置 leader) */
#define PROTO_QUERY_RANKS "QUERY_RANKS"
#define PROTO_RANK_INFO "RANK_INFO"

/* LEASE <server-id> model=<name> [duration=<s>|permanent]\n
 * → OK LEASED ranks=<n> leader=<ip:port> peers=<ip1,ip2,...>\n | ERR no-rank\n
 * server 向 supervisor 租用该模型一组空闲 rank(标记忙, 防其他 server 使用)。
 * peers = 组内各段节点 IP(段号顺序, server 随 INFER 捎给 rank0 组织协作)。
 * 不带 duration = 请求级(request 策略, 推理完即 RELEASE);
 * duration=<s> = 定时租用(timed, 到期自动回池); permanent = 永久(直到 RELEASE 或 server 死)。 */
#define PROTO_LEASE "LEASE"

/* RELEASE <server-id>\n → OK\n
 * server 释放租用的 rank 组, supervisor 同步更新为空闲。 */
#define PROTO_RELEASE "RELEASE"

/* ---- 流式 token 帧(server → router 透传, rank → server) ---- */

/* T <len>\n<token utf8 bytes>: 生成的一个 token */
#define PROTO_TOKEN "T"

/* DONE <gen_tokens> <eos=0|1> <ms>\n: 生成结束。
 * 会话模式时 server 会在末尾追加 <prompt_tokens>(HTTP usage 统计用)。 */
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
