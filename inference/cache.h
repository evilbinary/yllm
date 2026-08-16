#ifndef YLLM_CACHE_H
#define YLLM_CACHE_H

#include "yllm.h"

/* 会话缓存: key → 已缓存 token 序列(前缀缓存本体)。
 * 纯逻辑层: 不依赖引擎, 由 server(router)做会话管理时使用;
 * CLI 会话功能同样可复用。 */
typedef struct {
    char key[64];        /* 会话 key */
    uint32_t* tokens;    /* 已缓存完整 token 序列 */
    uint32_t n;          /* 已缓存长度 */
    uint32_t cap;        /* 容量 */
    uint64_t last_use;   /* LRU 时间戳 */
} SessVal;

typedef struct {
    SessVal* v;
    int n;               /* 条目数 */
    int cap;             /* 条目容量 */
} SessCache;

void sess_init(SessCache* c, int cap);
void sess_free(SessCache* c);

/* 查找 key(命中刷新 LRU), 返回条目或 NULL */
SessVal* sess_get(SessCache* c, const char* key);
/* 新建条目(已存在则返回已有), 超容量时 LRU 淘汰 */
SessVal* sess_put(SessCache* c, const char* key);
/* 请求 token 与缓存序列的公共前缀长度(逐 token 比对) */
uint32_t sess_prefix(const SessVal* v, const uint32_t* req, uint32_t n);
/* 前缀缩短(历史被改时截断到公共前缀) */
void sess_truncate(SessVal* v, uint32_t n);
/* 追加一批 token 到条目末尾, 返回 0 成功 */
int sess_commit(SessVal* v, const uint32_t* tok, uint32_t n);
/* 追加单个 token(生成流逐 token 记账), 返回 0 成功 */
int sess_append(SessVal* v, uint32_t tok);
/* LRU 淘汰最久未用的条目 */
void sess_evict(SessCache* c);

/* ---- 磁盘落盘 ---- */

/* 会话 token 列表落盘/载入(server 侧会话表持久化)。
 * 格式: magic(8B) + n_tokens(4B) + tokens(4B × n)。
 * 返回 0 成功; 载入时 v 自动扩容。 */
int sess_save(const SessVal* v, const char* path);
int sess_load(SessVal* v, const char* path);

/* 引擎 KV 缓存落盘/载入(rank 侧会话换出/恢复)。
 * 直接读引擎公开字段(e->kv/kv_dim/max_seq + ws.model.h.n_blocks), 引擎零改动。
 * 格式: magic(8B) + n_blocks/kv_dim/max_seq(各4B) + pos(4B)
 *       + 每层 K[0..pos] f16 + V[0..pos] f16
 * sess_kv_load 校验结构与引擎不符时返回 -1 且不写 pos。 */
int sess_kv_save(Engine* e, uint32_t pos, const char* path);
int sess_kv_load(Engine* e, const char* path, uint32_t* pos);

/* 生成缓存文件路径: <dir>/<安全化key><ext>。
 * key 中的 ':' 等文件系统非法字符替换为 '_'(NTFS 等不支持)。 */
void cache_path(char* out, size_t outsz, const char* dir, const char* key, const char* ext);

#endif /* YLLM_CACHE_H */
