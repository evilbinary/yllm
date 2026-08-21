/* config.h — 统一配置类(rank / server / router / supervisor / hub 共用)
 *
 * 所有命令的入参统一解析进 ServeConfig(命令行 --key value, 或 --config yaml),
 * 各角色从 config 取值初始化自己的结构, 不再各写各的 opt 解析。
 *
 * 优先级: 默认值 < yaml 文件(--config) < 命令行散参
 */

#ifndef YLLM_SERVE_CONFIG_H
#define YLLM_SERVE_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#define CFG_STR_MAX 1024

#define CFG_MAX_MODELS 8
typedef struct {
    char name[CFG_STR_MAX];   /* 模型注册名(router 路由用) */
    char model[CFG_STR_MAX];  /* llf 路径 */
    char vocab[CFG_STR_MAX];  /* vocab 路径 */
    char peers[CFG_STR_MAX];  /* 组内各段节点 IP(逗号分隔, 段号顺序; worker 段 spawn 时下发) */
    int  ranks;               /* 该模型的 rank 段数(总) */
    int  local;               /* 本机拉起的段数(默认 = ranks; 0 = 全部外部; 1 = 只 rank0 本地) */
    int  dist_fp16;           /* 段间激活 fp16 传输(带宽减半, 引入量化误差) */
} ModelCfg;

typedef struct {
    /* 身份/日志 */
    char node_id[CFG_STR_MAX];
    char role[32];              /* rank|server|router|supervisor|hub */
    char log_file[CFG_STR_MAX];
    char log_level[CFG_STR_MAX];
    int  no_console;

    /* 模型(单模型兼容: 顶层字段 = models[0]; 多模型用 models: 列表) */
    char model[CFG_STR_MAX];        /* llf 路径(= models[0].model) */
    char vocab[CFG_STR_MAX];        /* vocab 路径 */
    char model_name[CFG_STR_MAX];   /* 注册名(router 路由用) */
    char cache_dir[CFG_STR_MAX]; /* 会话缓存落盘目录(空 = 纯内存) */
    char bin[CFG_STR_MAX];          /* yllm 二进制路径 */
    ModelCfg models[CFG_MAX_MODELS];
    int n_models;

    /* 端口 */
    int sv_port;            /* supervisor 心跳口   9500 */
    int router_port;        /* router 客户端口     9400 */
    int server_port;        /* server 转发口       9420 */
    int rank_port_base;     /* rank 推理口基址     9410 */
    int http_port;          /* OpenAI HTTP         8000 */

    /* 拓扑 */
    char sv_host[CFG_STR_MAX];      /* supervisor 地址(rank/server 心跳目标) */
    char router_addrs[CFG_STR_MAX]; /* router 列表(supervisor 通知目标) */
    char leader[CFG_STR_MAX];       /* server 的 leader rank 地址 ip:port */
    int  ranks;                     /* 每 server 的 rank 段数 */
    int  servers;                   /* server 副本数 */

    /* 行为 */
    int  auto_heal;
    char strategy[CFG_STR_MAX];

    /* 租用(server) */
    char lease_strategy[CFG_STR_MAX];   /* request(用完即释放) | timed | permanent */
    int  lease_duration;                /* timed 策略的租期(秒) */

    /* rank 协作(多段) */
    int  rank_idx;                      /* --rank 段号(0..ranks-1) */
    char peers[CFG_STR_MAX];            /* 组内各段节点 IP(逗号分隔; worker 段命令行下发) */
    int  dist_fp16;                     /* 段间激活 fp16 传输(带宽减半, 引入量化误差) */

    /* 推理参数(rank 用) */
    float temp;
    float top_p;
    uint64_t seed;
    int64_t budget;         /* rank KV 内存预算(MB; -1 = auto)。解析时由 "auto"/"1024MB"/"1.5G" 转成 */
    int depth;
    char device[32];        /* cpu | cuda(见 docs/design-gpu-inference.md) */
    int gpu;                /* CUDA device index(默认 0) */
    char gpu_weights[16];   /* auto | q4k | fp16(CUDA 线性权上卡格式) */

    /* 客户端模式(router --send) */
    char send[CFG_STR_MAX];

    /* API 日志(router HTTP 请求/响应打印) */
    int  api_log;               /* 1 = 打印(默认) 0 = 关闭 */

    /* OpenAI 兼容 API key(Authorization: Bearer <key>; 空 = 不校验) */
    char api_key[CFG_STR_MAX];

    /* hub/supervisor 按模型名筛选(多模型 serve.yaml 只拉起指定模型; 逗号分隔多个) */
    char only_model[CFG_STR_MAX];
} ServeConfig;

/* 确保目录存在(逐级创建, 支持 "/" 与 "\\" 分隔)。返回 0 成功 */
static inline int config_ensure_dir(const char* path)
{
    if (!path || !path[0]) return 0;
    char buf[1024];
    size_t n = strlen(path);
    if (n >= sizeof(buf)) return -1;
    memcpy(buf, path, n + 1);
    size_t i;
    for (i = 0; buf[i]; i++) {
        if (buf[i] == '/' || buf[i] == '\\') {
            if (i == 0) continue;
            buf[i] = '\0';
#ifdef _WIN32
            if (_mkdir(buf) != 0 && errno != EEXIST) return -1;
#else
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
#endif
            buf[i] = '/';
        }
    }
#ifdef _WIN32
    if (_mkdir(buf) != 0 && errno != EEXIST) return -1;
#else
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return -1;
#endif
    return 0;
}

/* 启动时确保 cache_dir 存在(不存在则创建) */
static inline void config_ensure_cache_dir(const ServeConfig* c)
{
    if (c->cache_dir[0]) config_ensure_dir(c->cache_dir);
}

/* 填默认值 */
static inline void config_defaults(ServeConfig* c)
{
    memset(c, 0, sizeof(*c));
    {
        int mi;
        for (mi = 0; mi < CFG_MAX_MODELS; mi++) c->models[mi].local = -1; /* -1 = 未指定(默认全拉) */
    }
    snprintf(c->node_id, sizeof(c->node_id), "%s", "node-0");
    snprintf(c->log_file, sizeof(c->log_file), "%s", "logs/serve.log");
    snprintf(c->log_level, sizeof(c->log_level), "%s", "info");
    snprintf(c->vocab, sizeof(c->vocab), "%s", "vocab.txt");
    snprintf(c->model_name, sizeof(c->model_name), "%s", "default");
    snprintf(c->bin, sizeof(c->bin), "%s", "./build/avx2/yllm");
    snprintf(c->sv_host, sizeof(c->sv_host), "%s", "127.0.0.1");
    snprintf(c->strategy, sizeof(c->strategy), "%s", "least");
    snprintf(c->lease_strategy, sizeof(c->lease_strategy), "%s", "request");
    c->sv_port = 9500;
    c->router_port = 9400;
    c->server_port = 9420;
    c->rank_port_base = 9410;
    c->http_port = 8000;
    c->ranks = 1;
    c->servers = 1;
    c->temp = 1.0f;
    c->top_p = 0.9f;
    c->seed = 42;
    c->budget = -1;   /* 默认自动 */
    c->depth = 2;
    snprintf(c->device, sizeof(c->device), "cpu");
    c->gpu = 0;
    snprintf(c->gpu_weights, sizeof(c->gpu_weights), "auto");
    c->api_log = 1;   /* 默认开启 */
}

/* 解析内存预算字符串 → MB 数。
 * "auto" / 空 → -1(自动); "1024MB" / "1024m" → 1024; "1.5G" → 1536; "512KB" → 0(舍入)。
 * 无单位纯数字按 MB。 */
static inline int64_t config_budget_parse(const char* s)
{
    if (!s || !s[0]) return -1;
    if (strncmp(s, "auto", 4) == 0) return -1;
    char* end = NULL;
    double v = strtod(s, &end);
    if (end == s) return -1;
    while (end && (*end == ' ' || *end == '\t')) end++;
    double mb;
    if (*end == 'g' || *end == 'G')      mb = v * 1024.0;
    else if (*end == 'm' || *end == 'M') mb = v;
    else if (*end == 'k' || *end == 'K') mb = v / 1024.0;
    else if (*end == 't' || *end == 'T') mb = v * 1024.0 * 1024.0;
    else                                 mb = v;   /* 无单位按 MB */
    return (int64_t)mb;
}

/* 按 key 填一个字段(字符串/数字)。返回 1 认识该 key */
static inline int config_set(ServeConfig* c, const char* key, const char* val)
{
    if (strcmp(key, "node-id") == 0 || strcmp(key, "id") == 0) {
        snprintf(c->node_id, sizeof(c->node_id), "%s", val);
    } else if (strcmp(key, "log") == 0 || strcmp(key, "log-file") == 0) {
        snprintf(c->log_file, sizeof(c->log_file), "%s", val);
    } else if (strcmp(key, "log-level") == 0) {
        snprintf(c->log_level, sizeof(c->log_level), "%s", val);
    } else if (strcmp(key, "no-console") == 0) {
        c->no_console = atoi(val);
    } else if (strcmp(key, "model") == 0) {
        snprintf(c->model, sizeof(c->model), "%s", val);
    } else if (strcmp(key, "vocab") == 0) {
        snprintf(c->vocab, sizeof(c->vocab), "%s", val);
    } else if (strcmp(key, "model-name") == 0 || strcmp(key, "server-model") == 0) {
        snprintf(c->model_name, sizeof(c->model_name), "%s", val);
    } else if (strcmp(key, "bin") == 0) {
        snprintf(c->bin, sizeof(c->bin), "%s", val);
    } else if (strcmp(key, "sv-port") == 0) {
        c->sv_port = atoi(val);
    } else if (strcmp(key, "router-port") == 0) {
        c->router_port = atoi(val);
    } else if (strcmp(key, "server-port") == 0) {
        c->server_port = atoi(val);
    } else if (strcmp(key, "rank-port-base") == 0 || strcmp(key, "rank-port") == 0) {
        c->rank_port_base = atoi(val);
    } else if (strcmp(key, "http-port") == 0) {
        c->http_port = atoi(val);
    } else if (strcmp(key, "cache-dir") == 0) {
        snprintf(c->cache_dir, sizeof(c->cache_dir), "%s", val);
    } else if (strcmp(key, "sv-host") == 0 || strcmp(key, "addr") == 0 ||
               strcmp(key, "supervisor") == 0) {
        /* supervisor 可能是 ip:port, 拆开存 host + port */
        const char* colon = strchr(val, ':');
        if (colon) {
            size_t hlen = (size_t)(colon - val);
            if (hlen >= sizeof(c->sv_host)) hlen = sizeof(c->sv_host) - 1;
            memcpy(c->sv_host, val, hlen);
            c->sv_host[hlen] = '\0';
            c->sv_port = atoi(colon + 1);
        } else {
            snprintf(c->sv_host, sizeof(c->sv_host), "%s", val);
        }
    } else if (strcmp(key, "router") == 0 || strcmp(key, "router-addrs") == 0) {
        snprintf(c->router_addrs, sizeof(c->router_addrs), "%s", val);
    } else if (strcmp(key, "leader") == 0 || strcmp(key, "server-leader") == 0) {
        snprintf(c->leader, sizeof(c->leader), "%s", val);
    } else if (strcmp(key, "ranks") == 0) {
        c->ranks = atoi(val);
    } else if (strcmp(key, "servers") == 0) {
        c->servers = atoi(val);
    } else if (strcmp(key, "auto-heal") == 0) {
        c->auto_heal = atoi(val);
    } else if (strcmp(key, "strategy") == 0) {
        snprintf(c->strategy, sizeof(c->strategy), "%s", val);
    } else if (strcmp(key, "lease-strategy") == 0) {
        snprintf(c->lease_strategy, sizeof(c->lease_strategy), "%s", val);
    } else if (strcmp(key, "lease-duration") == 0) {
        c->lease_duration = atoi(val);
    } else if (strcmp(key, "rank") == 0) {
        c->rank_idx = atoi(val);
    } else if (strcmp(key, "peers") == 0) {
        snprintf(c->peers, sizeof(c->peers), "%s", val);
    } else if (strcmp(key, "dist-fp16") == 0) {
        c->dist_fp16 = atoi(val);
    } else if (strcmp(key, "temp") == 0) {
        c->temp = (float)atof(val);
    } else if (strcmp(key, "top-p") == 0) {
        c->top_p = (float)atof(val);
    } else if (strcmp(key, "seed") == 0) {
        c->seed = (uint64_t)strtoull(val, NULL, 10);
    } else if (strcmp(key, "budget") == 0) {
        c->budget = config_budget_parse(val);
    } else if (strcmp(key, "depth") == 0) {
        c->depth = atoi(val);
    } else if (strcmp(key, "device") == 0) {
        snprintf(c->device, sizeof(c->device), "%s", val);
    } else if (strcmp(key, "gpu") == 0) {
        c->gpu = atoi(val);
    } else if (strcmp(key, "gpu-weights") == 0 || strcmp(key, "gpu_weights") == 0) {
        snprintf(c->gpu_weights, sizeof(c->gpu_weights), "%s", val);
    } else if (strcmp(key, "send") == 0) {
        snprintf(c->send, sizeof(c->send), "%s", val);
    } else if (strcmp(key, "only-model") == 0) {
        snprintf(c->only_model, sizeof(c->only_model), "%s", val);
    } else if (strcmp(key, "api-log") == 0) {
        c->api_log = atoi(val);
    } else if (strcmp(key, "api-key") == 0 || strcmp(key, "api_key") == 0) {
        snprintf(c->api_key, sizeof(c->api_key), "%s", val);
    } else {
        return 0;
    }
    return 1;
}

/* 读 yaml 文件(仅支持 "key: value" 行, # 注释) */
static inline void config_load_yaml(ServeConfig* c, const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) return;
    char buf[65536];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;

    int in_models = 0;
    char* save = NULL;
    char* line = strtok_r(buf, "\n", &save);
    while (line) {
        char* p = line;
        int indent = 0;
        while (*p == ' ' || *p == '\t') { p++; indent++; }
        if (*p == '#' || *p == '\r' || *p == '\0') { line = strtok_r(NULL, "\n", &save); continue; }
        size_t l = strlen(p);
        while (l > 0 && (p[l-1] == '\r' || p[l-1] == '\n')) p[--l] = 0;

        if (in_models) {
            if (indent == 0) {
                /* 顶格: 退出 models 段, 该行按普通 key 处理 */
                in_models = 0;
            } else {
                if (strncmp(p, "- ", 2) == 0) {
                    p += 2;
                    while (*p == ' ') p++;
                    if (c->n_models < CFG_MAX_MODELS) c->n_models++;
                }
                if (c->n_models > 0) {
                    char* colon = strchr(p, ':');
                    if (colon) {
                        *colon = 0;
                        char* key = p;
                        char* val = colon + 1;
                        while (*val == ' ' || *val == '\t') val++;
                        char* hash = strchr(val, '#');
                        if (hash) *hash = 0;
                        size_t vl = strlen(val);
                        while (vl > 0 && (val[vl-1] == ' ' || val[vl-1] == '\t' || val[vl-1] == '\r')) val[--vl] = 0;
                        if (vl >= 2 && (val[0] == '"' || val[0] == '\'') && val[vl-1] == val[0]) {
                            memmove(val, val + 1, vl - 2);
                            val[vl - 2] = '\0';
                        }
                        ModelCfg* mc = &c->models[c->n_models - 1];
                        if (strcmp(key, "name") == 0) snprintf(mc->name, sizeof(mc->name), "%s", val);
                        else if (strcmp(key, "model") == 0) snprintf(mc->model, sizeof(mc->model), "%s", val);
                        else if (strcmp(key, "vocab") == 0) snprintf(mc->vocab, sizeof(mc->vocab), "%s", val);
                        else if (strcmp(key, "ranks") == 0) mc->ranks = atoi(val);
                        else if (strcmp(key, "local") == 0) mc->local = atoi(val);
                        else if (strcmp(key, "dist-fp16") == 0) mc->dist_fp16 = atoi(val);
                    }
                }
                line = strtok_r(NULL, "\n", &save);
                continue;
            }
        }

        /* 普通 key: value */
        {
            char* colon = strchr(p, ':');
            if (!colon) { line = strtok_r(NULL, "\n", &save); continue; }
            *colon = 0;
            char* key = p;
            char* val = colon + 1;
            while (*val == ' ' || *val == '\t') val++;
            char* hash = strchr(val, '#');
            if (hash) *hash = 0;
            size_t vl = strlen(val);
            while (vl > 0 && (val[vl-1] == ' ' || val[vl-1] == '\t' || val[vl-1] == '\r')) val[--vl] = 0;
            /* 剥离 YAML 字符串两侧引号 */
            if (vl >= 2 && (val[0] == '"' || val[0] == '\'') && val[vl-1] == val[0]) {
                memmove(val, val + 1, vl - 2);
                val[vl - 2] = '\0';
            }
            if (strcmp(key, "models") == 0) in_models = 1;
            else config_set(c, key, val);
        }
        line = strtok_r(NULL, "\n", &save);
    }

    /* 兼容: 无 models 列表时, 顶层单模型字段填入 models[0] */
    if (c->n_models == 0 && c->model[0]) {
        c->n_models = 1;
        snprintf(c->models[0].name, sizeof(c->models[0].name), "%s",
                 c->model_name[0] ? c->model_name : "default");
        snprintf(c->models[0].model, sizeof(c->models[0].model), "%s", c->model);
        snprintf(c->models[0].vocab, sizeof(c->models[0].vocab), "%s", c->vocab);
        c->models[0].ranks = c->ranks > 0 ? c->ranks : 1;
    }
    /* 反向: models[0] 同步到顶层(rank/server 直接读顶层字段) */
    if (c->n_models > 0) {
        if (c->models[0].model[0]) snprintf(c->model, sizeof(c->model), "%s", c->models[0].model);
        if (c->models[0].vocab[0]) snprintf(c->vocab, sizeof(c->vocab), "%s", c->models[0].vocab);
        if (c->models[0].name[0]) snprintf(c->model_name, sizeof(c->model_name), "%s", c->models[0].name);
        if (c->models[0].ranks > 0) c->ranks = c->models[0].ranks;
    }
}

/* 命令行散参: --key value 或 --key=value, 覆盖到 config */
static inline void config_load_args(ServeConfig* c, int argc, char** argv, int start)
{
    int i;
    for (i = start; i < argc; i++) {
        const char* a = argv[i];
        if (a[0] != '-' || a[1] != '-') continue;
        const char* key = a + 2;
        const char* val = NULL;
        const char* eq = strchr(key, '=');
        char keybuf[128];
        if (eq) {
            size_t klen = (size_t)(eq - key);
            if (klen >= sizeof(keybuf)) klen = sizeof(keybuf) - 1;
            memcpy(keybuf, key, klen);
            keybuf[klen] = '\0';
            key = keybuf;
            val = eq + 1;
        } else if (i + 1 < argc) {
            val = argv[i + 1];
            i++;
        }
        if (!val) continue;
        if (strcmp(key, "config") == 0) continue;   /* --config 单独处理 */
        /* 角色端口映射: --port 映射到角色端口(由 --role 指定) */
        if (strcmp(key, "port") == 0) {
            const char* role = c->role[0] ? c->role : "supervisor";
            if (strcmp(role, "server") == 0) c->server_port = atoi(val);
            else if (strcmp(role, "router") == 0) c->router_port = atoi(val);
            else if (strcmp(role, "rank") == 0) c->rank_port_base = atoi(val);
            else c->sv_port = atoi(val);
            continue;
        }
        config_set(c, key, val);
    }
}

/* 按角色修正端口: --port 是通用参数, 映射到角色端口 */
static inline void config_apply_role(ServeConfig* c, const char* role)
{
    if (strcmp(role, "server") == 0) {
        /* --port 已存 sv_port, 若用户没显式给 server-port 则视为 server 端口 */
    } else if (strcmp(role, "router") == 0) {
    } else if (strcmp(role, "rank") == 0) {
    }
    /* supervisor: sv_port 即自身端口 */
}

/* 统一加载: 默认值 → --config yaml → 命令行散参 */
/* 多模型端口/ID 步长 = 所有模型里最大的 ranks(避免端口重叠; 无 16/32 魔数限制) */
static inline int cfg_model_stride(const ServeConfig* c)
{
    int stride = 1, mi;
    for (mi = 0; mi < c->n_models && mi < CFG_MAX_MODELS; mi++)
        if (c->models[mi].ranks > stride) stride = c->models[mi].ranks;
    return stride;
}

/* 按模型名筛选(models 列表只保留名字在逗号分隔列表中的; 命中 0 个则保持原样)。
 * 供 hub / supervisor 用 --only-model 只拉起指定模型, 避免一次带起全部。 */
static inline void config_filter_models(ServeConfig* c, const char* names)
{
    if (!names || !names[0] || c->n_models == 0) return;
    ModelCfg kept[CFG_MAX_MODELS];
    int nk = 0, mi;
    for (mi = 0; mi < c->n_models; mi++) {
        const char* p = names;
        int hit = 0;
        while (*p) {
            const char* comma = strchr(p, ',');
            size_t len = comma ? (size_t)(comma - p) : strlen(p);
            if (len > 0 && strncmp(c->models[mi].name, p, len) == 0 &&
                c->models[mi].name[len] == '\0') { hit = 1; break; }
            p = comma ? comma + 1 : p + strlen(p);
        }
        if (hit) kept[nk++] = c->models[mi];
    }
    if (nk == 0) return; /* 没命中任何名字, 视为不筛选 */
    c->n_models = nk;
    for (mi = 0; mi < nk; mi++) c->models[mi] = kept[mi];
}

static inline void config_load(ServeConfig* c, int argc, char** argv, int start)
{
    config_defaults(c);
    /* 角色从 argv[1] 推断 */
    if (start >= 1 && argv[0]) {
        /* argv[0] 是程序名, 角色在 argv[1] */
    }
    if (argc > 1 && argv[1]) snprintf(c->role, sizeof(c->role), "%s", argv[1]);
    int i;
    const char* cfg_path = NULL;
    for (i = start; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            cfg_path = argv[i + 1];
            break;
        }
    }
    /* 未指定 --config 时默认 serve.yaml(存在才加载) */
    if (!cfg_path) {
        FILE* t = fopen("serve.yaml", "r");
        if (t) { fclose(t); cfg_path = "serve.yaml"; }
    }
    if (cfg_path) config_load_yaml(c, cfg_path);
    config_load_args(c, argc, argv, start);
}

#endif /* YLLM_SERVE_CONFIG_H */
