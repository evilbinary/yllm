# yllm 常驻推理服务架构设计(router / server / 公用 rank 池 / supervisor)

版本:v0.3 ｜ 状态:设计 ｜ 目标:模型权重常驻,进程活等请求,调度层只做毫秒级路由决策

## 0. 结论摘要

- **核心问题**:当前 `dist-worker` 每次 `run` 都 `spawn_gen` 拉起一个全新 `yllm gen` 进程,
  推理完即退出。每次推理 = 一次完整冷启动(mmap 模型、分配 KV、加载 tokenizer、分布式握手),
  无常驻、无路由、无请求队列。
- **方案**:四层模型 + 一个统一生命周期管理者:

```
router(路由) ──▶ server(业务逻辑组) ──▶ 公用 rank 池(计算单元) ──运行在──▶ 机器
   │                      │                     │                        ▲
   │ 只决定请求用哪个server  │ 租用 rank 组成流水线   │ 通用可互换的推理单元     │
   └──────────────────────┴─────────────────────┴────────────────────────┘
                                          supervisor(唯一管所有进程生死)
```

- **router**:无状态、多实例,只做路由决策(模型过滤 → 会话亲和 → 负载),对外提供 **OpenAI 兼容 HTTP 接口**;
- **server**:独立常驻进程,业务逻辑组(请求队列、会话亲和、从公用 rank 池租用 rank 组成流水线);一台机器可起多个实例(不同模型,或同模型多副本共享 mmap 页缓存);
- **rank**:公用的调度最小单位,分布在多台机器,常驻推理进程(一个层段),可被任意 server 租用;
- **supervisor**:管机器清单 + 全部进程(router/server/rank)生命周期(拉起/回收/模型更新/自愈/扩缩容)。
- 进程只在该重启时重启(部署更新/扩缩容/故障自愈/显存整理),与"每次请求"完全解耦。
- 落地顺序:P1 单 rank 常驻 `serve` + 帧协议 → P2 router 多 server 路由 + 租用 rank 组 → P3 supervisor 生命周期 + OpenAI 接口。
- 复用已有资产:私有 TCP 帧风格(`dist.c`)、llf mmap 权重加载(`engine_init`)、
  分布式层流水线(`dist_split_layers`/`dist_gen`)、`log.c` 日志、`dist-worker` 常驻骨架。

## 1. 现状与问题

```
客户端 ──▶ dist-worker(控制口 9355)
            └─ run 命令 ──▶ spawn_gen ──▶ 新 yllm gen 进程
                 ├─ mmap 模型(冷启动 30-60s)   ← 每次请求重复
                 ├─ vocab_encode → dist_gen → 推理
                 └─ 进程退出(权重/缓存全释放)   ← 下次再来一遍
```

问题清单:

| # | 问题        | 影响                                                |
| - | --------- | ------------------------------------------------- |
| 1 | 每次推理都新起进程 | 冷启动延迟 30-60s,不可用于交互                               |
| 2 | 进程用完即退    | 无 KV/prefix cache 可复用,长会话零复用                      |
| 3 | 无请求队列     | 并发请求要么串行 spawn 多进程,要么丢                            |
| 4 | 无健康/就绪探测  | 拉起来就当可用,无 liveness/readiness 区分                   |
| 5 | 调度与生命周期耦合 | worker 既 spawn 又路由,无法做滚动更新/扩缩容                    |
| 6 | 命令形态混乱    | `dist-worker` 用 `--serve/--port/--send` 区分模式,不够清晰 |

## 2. 目标架构

### 2.1 分层模型

```
                         ┌────────────────────────────────────────┐
   客户端(OpenAI API) ──▶ │ router(多实例, 无状态, supervisor 管)   │
                         │  ① 模型过滤 ② 会话亲和 ③ 负载           │
                         └──────────────┬─────────────────────────┘
                            ┌───────────┴───────────┐
                            ▼                       ▼
                 ┌─────────────────┐      ┌─────────────────┐
                 │ server-q1 (qwen3)│      │ server-t1 (tinyllama)│
                 │ server-q2 (qwen3)│      │ server-t2 (tinyllama)│
                 │ 业务逻辑+会话亲和   │      │ 业务逻辑+会话亲和   │
                 │ 租用 rank 组       │      │ 租用 rank 组       │
                 └───────┬─────────┘      └────────┬─────────┘
                         ▼                          ▼
                  ┌────────────────────────────────────────────┐
                  │           公用 rank 池(跨机器)                │
                  │   qwen3 段 rank / tinyllama 段 rank 混合      │
                  └────────────────────────────────────────────┘
                         ▲
                  ┌──────┴─────────────────────────────────────┐
                  │ supervisor: 管机器清单 + 全部进程生命周期      │
                  │  (router/server/rank 拉起、回收、更新、自愈)   │
                  └────────────────────────────────────────────┘
```

### 2.2 角色边界(对齐"调度层不触碰进程生命周期"原则)

| 层          | 是否进程            | 调度/决策                               | 生命周期                    |
| ---------- | --------------- | ----------------------------------- | ----------------------- |
| router     | 常驻,多实例          | 路由:这个请求发给哪个 server(毫秒查表)            | supervisor 管            |
| server     | 常驻,每业务组一个,可同机多个 | 调度:租用 rank 组成流水线,把任务交给组内 leader     | supervisor 管            |
| rank       | 常驻,机器上若干        | 推理单元:一个层段,组内流水线协作,通用可互换             | supervisor 管(经机器 agent) |
| supervisor | 常驻(管理节点)        | 决策:哪台机器、起几个 rank/server/router、各跑什么 | 唯一管进程生死                 |
| 机器         | 物理              | 承载进程,agent 执行 supervisor 指令         | supervisor 管理清单         |

### 2.3 调度方向(单向,不越级)

```
router ──▶ server    (路由: 选哪个业务组)
server ──▶ rank      (调度: 租用哪组 rank 拼流水线)
rank   ──▶ 机器      (运行: 分布在哪些机器)
supervisor ──▶ 机器 + rank + server + router  (生命周期: 拉起/回收/更新/自愈)
```

- router 不碰 rank:只看到 server 列表;
- server 不碰机器:只从池里选 rank,不关心 rank 在哪台机器;
- rank 通用:同模型 rank 同构可互换,谁租到谁用;
- supervisor 不碰请求:只维护进程生死。

### 2.3.1 状态归属(谁报、谁用、谁判死)

| 状态             | 谁产生/上报                  | 谁消费                 | 谁判定/处理                                 |
| -------------- | ----------------------- | ------------------- | -------------------------------------- |
| 启动(LOADING)    | rank 自身(PING 回 LOADING) | server / supervisor | —                                      |
| 就绪(READY)      | rank 自身(PING 回 READY)   | server / supervisor | —                                      |
| 忙/空闲(inflight) | rank 自身(STAT)           | server(租用/调度决策)     | —                                      |
| KV/prefix 占用   | rank 自身(STAT)           | server(路由决策)        | —                                      |
| rank DEAD      | —                       | server 上报           | **supervisor**(重拉)                     |
| server DEAD    | —                       | router(心跳) → 上报     | **supervisor**(回收租约→rank 回池→重拉 server) |
| router DEAD    | —                       | supervisor          | **supervisor**(重拉,靠心跳自愈注册表)            |

一句话:rank 报状态、server 用状态、supervisor 管死活(含租约回收与 rank 池权威)。

### 2.3.2 server 生命周期边界

- server 是**租用者**不是所有者:rank 归属 supervisor 池,server 死了 rank 回池,其他 server 可再租;
- server **感知负载**并上报 `SCALE`,但**永不 spawn/kill rank**(生命周期统一归 supervisor);
- server 自己也是被 supervisor 管理的进程(挂了重拉,重新从池租 rank)。

### 2.4 何时重启进程(仅四类,均与"每次调度"无关)

1. **部署更新**:换模型/配置/二进制 → 滚动更新(起新 → 就绪 → 摘旧 → DRAIN 停);
2. **弹性扩缩容**:增删 server/rank 实例(新增=冷启动;缩容=优雅 DRAIN);
3. **故障自愈**:OOM/崩溃 → supervisor 检测 dead → 按原配置重拉;
4. **显存碎片整理**:长稳后吞吐下降 → 逐个滚动重启(周级别一次)。

## 3. 组件设计

### 3.1 命令形态(收拢到 `yllm` 子命令)

| 命令                | 角色     | 进程形态          | 职责                                           |
| ----------------- | ------ | ------------- | -------------------------------------------- |
| `yllm rank`       | 公用推理单元 | 常驻(每机器每模型段一个) | 加载层段权重、组内流水线协作、收 INFER/PING/STAT/DRAIN 帧     |
| `yllm server`     | 业务逻辑组  | 常驻(每业务组一个)    | 租用 rank 组、接请求入队、会话亲和、转发给 leader、流式汇总         |
| `yllm router`     | 调度层    | 常驻(多实例)       | server 注册表、心跳、路由决策、OpenAI 兼容 HTTP、请求转发       |
| `yllm supervisor` | 管理节点   | 常驻(管理机一个)     | 管机器 + 全部进程生命周期 + **文件分发(serve/sync)** + 模型更新 |

说明:`dist-worker --serve`(模型/日志文件中转)不再单独立进程,并入 `yllm supervisor`;
rank/server/router 的二进制与模型由 supervisor 先分发(sync)到目标机器,再拉起进程。

### 3.2 rank(公用资源池单元,`yllm rank` 常驻)

**启动阶段(一次)**

1. 解析 `--model --vocab --rank --ranks --port-base --dist-addrs --budget-mb --depth` 等;
2. `engine_init`(mmap 模型) → `dist_split_layers`(领自己的层段) → `vocab_load`;
3. `dist_init` 与组内兄弟 rank 建立常驻 TCP(流水线握手一次,不复用后销毁);
4. 监听推理端口,进入事件循环。

**请求阶段(循环,层段权重常驻)**

- 收帧 → `vocab_encode`(仅 rank0)→ 本地算自己层段 → 经 dist 与组内兄弟协作 → 流式写回 token → 回完成统计;返回后不退出。
- 组内维护:请求队列、当前 inflight、KV/prefix cache 占用统计。

**帧协议(私有 TCP,一行命令 + 二进制 payload,风格沿用 dist.c)**

| 命令                                                | 请求                                          | 响应                                                         | 说明                 |
| ------------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------- | ------------------ |
| `PING`                                            | `PING\n`                                    | `OK READY\n` / `OK LOADING\n` / `ERR <msg>`                | liveness/readiness |
| `STAT`                                            | `STAT\n`                                    | `OK inflight=<n> kv_mb=<f> prefix_hits=<n> uptime_s=<t>\n` | 供 server 路由决策      |
| `INFER <max_tokens> <n_bytes>\n` + `prompt bytes` | 流式 `T <len>\n` + bytes ... `DONE <stats>\n` | 生成请求,流式 token                                              | <br />             |
| `DRAIN`                                           | `DRAIN\n`                                   | `OK\n`(等当前请求完) 后关连接退出                                      | 优雅下线               |
| `QUIT`                                            | `QUIT\n`                                    | `OK\n` 后退出                                                 | 强制退出(仅测试)          |

设计要点:

- INFER 的 prompt 用长度前缀(binary),避免空格/换行歧义;token 流式回 `T` 帧。
- rank 默认**串行**推理(CPU 模型),请求排队;并发作为后续优化(多 worker 线程共读 mmap)。
- rank 只关心自己那一段层 + 与上下级 rank 的 TCP,不感知全局。

**rank 伪代码流程**

```
yllm rank (模型段, 常驻进程)
────────── 初始化(一次) ──────────
  engine_init(model)                # mmap 权重
  dist_split_layers(rank, ranks)    # 领自己的层段
  vocab_load()
  if 多 rank: dist_init(...)        # 与兄弟 rank 建常驻 TCP
  listen(推理端口)

────────── 主循环 ──────────
loop:
  frame = recv()
  switch frame.cmd:
    PING:   reply OK (LOADING | READY)   # 存活+就绪
    STAT:   reply inflight / kv_mb / prefix_hits
    INFER:  handle_infer(frame)          # 只有 rank0 会收到
    DRAIN:  stop_accept(); drain_remaining(); exit()
    QUIT:   exit()

handle_infer(frame):                     # rank0 入口
  enqueue(prompt, max_tokens, ...)
  ids = vocab_encode(prompt)
  while 未到 max_tokens / 未 eos:
    out = forward_range(ids[pos])        # 算自己层段
    if rank < ranks-1: dist_send_x(out)  # 发给下游
    else:              # 末 rank
      logits = final_norm + lm_head(out)
      dist_send_logits(rank0)
    if rank == 0:
      id = sample(logits)
      reply T <token>                    # 流式回 server
  reply DONE <stats>
  inflight--
```

> rank 只算自己的层段 + 收发激活帧;`PING/STAT/INFER/DRAIN` 是服务层帧(protocol.h),激活帧是 dist 内部帧。

### 3.3 server(业务逻辑组,`yllm server` 独立常驻进程)

**组成与生命周期**

- 一个 server = **一个或多个 rank 组**(每组 rank0..rank n,可跨机器),由 supervisor 从公用 rank 池租用;
- 一台机器可起多个 server 实例:不同模型(各自独立权重),或同模型多副本(共享 mmap 页缓存);
- 请求进入 server → 入队 → 分给某组的 leader rank0 → 流水线协作 → 流式汇总回 router。

**server 可负责多组 rank(组 = 并发粒度)**

```
server-q1(业务逻辑组)
  ├─ 会话亲和表: session → 组
  ├─ 请求队列
  ├─ 组 0: rank0a..rankNa   ← 租用组1(处理部分会话)
  ├─ 组 1: rank0b..rankNb   ← 租用组2(另一部分会话)
  └─ 组 2: rank0c..rankNc   ← 可再扩(SCALE 加的)
```

- **组内串行,组间并行**:同一组 rank 同一时刻处理一个请求(一条流水线);不同组各自独立流水线并行;
- **server 并发度 = 它租用的组数**:请求多 → 上报 SCALE → supervisor 加一组 → server 并发 +1;
- 组内分配(会话亲和落到组):同 session 钉同一组(复用该组 KV/prefix);新 session 或原组过载 → 挑 least-inflight 的组;
- "组"是计算并发的粒度,"server"是业务/配置/生命周期的粒度——一个 server 可挂 1..N 组;
- 需要独立更新/隔离/不同模型 → 拆成多个 server。

**一次请求的数据流(server 只连 leader rank0)**

```
router ──INFER──▶ server
                     │  派发给 leader(组内唯一对外入口)
                     ▼
                 rank0
                 ├─ vocab_encode(prompt)
                 ├─ 算自己层段 → 激活 ──▶ rank1 ──▶ ... ──▶ rankN
                 │                              (层段接力)
                 ├─ 收 rankN 的 logits
                 ├─ 采样 → token
                 ├─ 流式回 server ◀── 实时, 每 token 一帧
                 └─ (下一 token 再用激活走一遍流水线)
                     │
                     ▼
server ──流式回──▶ router ──▶ 用户
```

- server **不连 rank1..N**,只连 leader rank0;rank0 负责组内通信与结果汇总;
- 组内任一 rank 挂 → 整组 DEAD(流水线断)→ 上报 supervisor 重拉整组。

**server 与 rank 是"租用(lease)"关系,不是"拥有"**

- rank 的生命周期与归属权在 supervisor(公用 rank 池的权威记录),server 只是**租用者**;
- server 从池租到 rank 组时登记租约;租约带心跳/超时;
- **server 死了 → 租约超时 → supervisor 回收 → rank 释放回公用池** → 其他 server 可从池再租(rank 不随 server 死而丢失);
- 其他 server 找 rank 通过 supervisor 维护的池,不看死掉的 server。

**server 内维护**

- 请求队列、当前 inflight、KV/prefix 占用(汇总各组内各 rank);
- **会话亲和表**:`session_id → 组`(同一会话钉在同一组,复用 KV/prefix)——会话亲和下沉到 server,router 因此无状态;
- 任一组内任一 rank 挂 → 该组 DEAD → 上报 supervisor 重拉该组。

**server 扩容不自 spawn rank,只上报需求**

- server 监控组内 inflight / 队列深度 / KV 占用;
- 超过阈值 → 向 supervisor 发 `SCALE`(当前 inflight、队列长度、建议目标组数);
- 由 supervisor 查机器清单决策并拉起新 rank 组 → 组进 server → 扩容完成;
- server 永远不 spawn/kill rank(生命周期统一归 supervisor)。

**server 帧(router ↔ server,私有 TCP)**

| 命令                                                     | 请求                                                           | 响应                   |
| ------------------------------------------------------ | ------------------------------------------------------------ | -------------------- |
| `REGISTER`                                             | `REGISTER server-a model=tinyllama leader=ip:port ranks=n\n` | `OK\n`(广播到所有 router) |
| `HEARTBEAT`                                            | `HEARTBEAT inflight=2 kv_mb=380\n`                           | `OK\n`(每 2s 广播)      |
| `INFER <session_id> <max_tokens> <n_bytes>\n` + prompt | 流式 `T` ... `DONE`                                            | 生成请求                 |
| `SCALE`                                                | `SCALE server-a need_groups=3\n`                              | `OK\n`(supervisor 处理后通知) |
| `DRAIN`                                                | `DRAIN\n`                                                    | `OK\n` 后退出(滚动更新/缩容)  |

**server 伪代码流程**

```
yllm server (业务组, 常驻进程, 从 rank 池租用一组或多组 rank)
────────── 初始化 ──────────
  REGISTER 到所有 router: (id, model, leader_addr, ranks)
  lease = supervisor 分配的 rank 组(可多组, 每组 rank0..rankN)

────────── 主循环 ──────────
loop:
  frame = recv()    # 来自 router 的请求 / supervisor 的管理
  switch:
    INFER(session_id, prompt, max_tokens): handle_infer()
    SCALE_ACK: 新 rank 组已就绪 → 加入组列表
    DRAIN:     drain_remaining(); 归还租约; exit()

handle_infer(req):
  # ① 会话亲和(落到组): 同 session 钉同一组, 复用该组 KV/prefix
  g = session_table[req.session_id]
  if g 过载: g = least_inflight(我的组们); session_table[sid] = g
  # ② 入队(组内串行)
  enqueue(g, req)
  # ③ 派发给该组 leader rank0(不连 rank1..N)
  resp = forward(g.leader, INFER)
  # ④ 流式转发回 router
  while resp 有 token: relay T → router
  relay DONE
  inflight--, kv_mb 更新

────────── 后台(并发) ──────────
every 2s:
  HEARTBEAT(inflight, kv_mb) → 所有 router   # 供路由决策
  if 队列深度 > 阈值 持续 10s:
      send SCALE(期望组数+1) → supervisor     # 只上报, 不自己拉 rank
  if 某组 rank PING 超时:
      上报 supervisor(回收租约 + 重拉该组)
```

> server 是"租用者":从 supervisor 拿 rank 组,挂了由 supervisor 回收租约、rank 回池。

### 3.4 router(调度层,`yllm router` 无状态多实例)

**OpenAI 兼容 HTTP 面(对外)**

| 端点                          | 说明                               |
| --------------------------- | -------------------------------- |
| `POST /v1/chat/completions` | chat,支持 `stream`(SSE)与 JSON 两种返回 |
| `POST /v1/completions`      | 文本补全                             |
| `GET /v1/models`            | 已注册模型列表                          |
| `GET /health`               | 存活/就绪,供 LB / supervisor 探测       |

请求/响应映射:

- `messages` 末条 user content → prompt;`max_tokens/temperature/top_p` → INFER 参数;
- `model` → 查注册表(模型 hash)→ 路由;流式用 SSE `data:` 帧,结束 `data: [DONE]`;
- 非流式返回 OpenAI 完整 JSON(`choices[].message.content` + `usage`)。

**内部业务路由(毫秒,无状态)**

```
router.route(req):
  ① 模型过滤:  candidates = [s for s in registry if s.model_hash == req.model]
  ② 会话亲和:  req.session_id 在表 → 该 server READY? → 直接发它(复用 KV/prefix)
               server DEAD → 删映射, 降级
  ③ 前缀亲和:  prompt hash 命中某 server 的 prefix cache → 发它
  ④ 负载:      least-inflight / kv-aware / round-robin → 选一个
```

**无状态化的关键**:

- server 注册表:server 启动/心跳时向**所有 router** 广播,router 间无需同步;
- 会话亲和:下沉到 server,router 不记会话;
- 决策纯函数,基于各自缓存的心跳数据 → 多实例天然成立,挂一个其他照常;
- 客户端可连任意 router(前面加 LB 或客户端轮询)。

**路由层铁律**:不 spawn 不 kill 进程、不加载模型。DEAD server 只剔除 + 上报,拉起是 supervisor 的事。

**router 伪代码流程**

```
yllm router (常驻, 多实例, 无共享状态)
────────── 注册表 ──────────
servers[server_id] = { model, leader_addr, state, inflight, kv_mb, last_hb }
   # 由各 server 心跳广播填充, 不主动拉取

────────── 主循环(HTTP 线程) ──────────
handle_http(req):                       # OpenAI 兼容
  target = route(req)
  if !target: reply 503
  resp = forward(target.leader, INFER)
  if stream: SSE(data: ...) else JSON

route(req):                             # 毫秒, 无状态
  candidates = [s for s in servers
                if s.model == req.model and s.state == READY]
  if empty: return NULL
  # ① 会话亲和(无状态, 一致性哈希): 同 session 钉同一 server(复用 KV)
  if req.session_id:
    t = candidates[hash(req.session_id) % len(candidates)]
    if t 未过载: return t
  # ② 负载策略:
  strategy = config (least_inflight | kv_aware | round_robin)
  return pick(candidates, strategy)

────────── 后台(心跳线程) ──────────
every 2s:
  for s in servers:
    if 超时未 HEARTBEAT: s.state = DEAD   # 只剔除, 不拉进程
  notify supervisor(DEAD list)
```

> router 无状态:会话亲和用一致性哈希(不存表),注册表靠 server 心跳广播自愈,挂实例由 supervisor 重拉。

### 3.5 supervisor(生命周期管理层,唯一管进程)

- **管机器清单**:每台机器的核数/内存/已部署进程 → 决定拓扑;
- **rank 池权威**:维护所有机器的 rank 注册表(谁在哪台、什么状态、被谁租用)——rank 归属权在 supervisor,不在 server;
- **租约管理**:server 租用 rank 组时登记 lease;server 死亡/租约超时 → 回收 rank 释放回公用池,其他 server 可再租;
- **文件分发(serve/sync)**:管理节点起文件服务,把模型 llf/二进制/配置经 sync 推送到各目标机器(复用现有私有 TCP 帧文件服务),拉起 rank 前先保证文件就位;远端更新日志同样经 sync 拉回;
- **拉起**:按模型切分(ranks)与副本数,决策"哪台机器起几个 rank/server/router",经机器 agent 执行;
- **扩缩容**:收 server 的 `SCALE` 请求 → 查机器清单 → 起新 rank 组(或 DRAIN 多余组)→ 通知 server;
- **一台机器几个 server 实例的决策**:
  ```
  同模型多副本 → 优先同机(共享 mmap 页缓存), 每副本只加 KV + 激活内存
  不同模型     → 各自独立权重, 分散放置
  数量上限     → min(核数÷每实例核, (内存-系统预留)÷每实例KV+激活)
  ```
- **模型更新(滚动更新, supervisor 唯一负责)**:
  ```
  supervisor:
    make serve-roll NEW_MODEL=test/v2.llf
    ├─ ① 部署新模型: 起 v2 的 rank 段 → 等 READY
    │              → 起 v2 的 server 实例 → 注册进 router(作为新 server)
    ├─ ② 流量切换: router 的注册表里加入 v2 server, 摘掉 v1 server(新请求不再路由到 v1)
    │             (router 侧由 supervisor 下发配置/或新 server 就绪后旧 server 注销)
    ├─ ③ 优雅下线 v1: 对 v1 server 发 DRAIN(等当前请求跑完) → 停进程
    │              → 回收 v1 租用的旧 rank(重拉为 v2 段或释放)
    └─ ④ 完成: 全部流量在 v2, v1 无残留进程
  ```
  注意:同一模型小版本热替换(v1.1→v1.2,结构不变)可复用已加载权重,只更新权重文件;
  结构变更(层数/vocab 变)必须整套 rank 重启。
- **进程自恢复**:轮询 router DEAD 列表 + 机器 agent 检测 → 重拉 router/server/rank;
- **扩缩容**:目标 server 数 N → 缺则租用新 rank 组,多则 DRAIN 多余组;
- **显存碎片整理**:长稳后吞吐下降 → 逐个滚动重启(周级别一次);
- 落地形态:先做 Makefile 目标,复杂编排留给 K8s。

**supervisor 伪代码流程**

```
yllm supervisor (管理节点, 常驻进程, 唯一 spawn/kill 进程的一方)
────────── 状态 ──────────
machines:   {ip → 核数/内存/已部署进程}
rank_pool:  {rank_id → 机器/模型段/状态/被谁租用}   # rank 池权威
processes:  {type → pid, state}                    # 全部进程
leases:     {server_id → rank 组}

────────── 主循环(事件驱动) ──────────
loop:
  evt = recv_event()
  switch:
    SCALE(server, 期望组数):
      if 机器清单有资源:
        for i in 需要的 rank 段数:
          选机器 → agent 拉起 rank → 等 PING READY
        新组加入 server.leases → 通知 server(SCALE_ACK)
      else: 拒绝, 通知 server 排队或降级

    DEAD(server/router/rank):           # 来自心跳检测
      if server: 回收 lease → rank 回 rank_pool → 按配置重拉 server
      if router: 重拉 router(靠心跳自愈注册表)
      if rank:   # 当前策略: 整组重建(见 3.5.1)
        回收该组全部 rank 租约 → 回 rank_pool
        → 从池重租一组 → 拉起/READY → 通知 server 更新 leader

    MODEL_UPDATE(new_model):
      ① 起 v2 rank 段 → READY
      ② 起 v2 server → REGISTER 到 router
      ③ router 摘 v1 server(新请求不再路由到它)
      ④ v1 server DRAIN(等当前请求完) → 停
      ⑤ 回收 v1 rank → 重拉为 v2 段或释放

    ROLL_UPDATE(整组滚动, 显存整理):
      for server in servers:
        起新 → 摘旧 → DRAIN 旧 → 停    # 逐个, 周级别一次

────────── 后台(健康检测) ──────────
every 3s:
  for p in processes:
    if PING 超时: p.state = DEAD; emit DEAD(p)
```

> supervisor 是**唯一** spawn/kill 进程的一方;rank 归属权在它;server 只上报 SCALE,不自己拉 rank。

### 3.5.1 rank 挂掉的重建策略

**策略:先整组重建,保证正确**(P1/P2 阶段 dist 仅启动时握手、无运行时重连)。

| 挂的位置 | 影响 | 当前策略(整组重建) | 后续优化(需 dist 支持运行时重连) |
|---|---|---|---|
| rank0 挂 | 入口+采样点没了,整条流水线无法起步 | 整组重建(唯一选择) | 整组重建(不可省) |
| 中间 rank 挂(如 rank1) | 流水线在中间断,rank0 发激活失败,下游空等 | 整组重建 | 只替换该 rank,重连握手 |
| 末 rank 挂 | 收不到 logits,rank0 无法采样 | 整组重建 | 只替换末 rank |

**整组重建流程(supervisor)**

```
组 G0 中任一 rank 挂:
  ① 回收 G0 全部 rank 租约 → 全部释放回公用池
     (挂掉的丢, 活着的也没用 —— 流水线已断, 单段无法推理)
  ② 从池重新租一组 G0'(新 rank0'..rankN')
     → 拉起/等 READY → 流水线握手
  ③ 通知 server-q1: 新组 G0' 就绪, 更新 leader 地址
  ④ 会话: KV 全部丢失, 客户端需重发完整上下文(从头 prefill)
  ⑤ 进行中的请求失败 → 客户端重试
```

**为什么先整组重建**
- 当前 `dist` 只在启动时 `dist_init` 握手一次,无"邻居断开→等待→重新握手"的运行时重连;
- 只替换单 rank 需要:新 rank 与邻居重新握手 + 流水线重新对齐 + 各 rank KV 重填,复杂度高;
- 整组重建逻辑与 SCALE/MODEL_UPDATE 复用同一套"租组→拉起→READY→通知"流程,简单可靠。

**后续优化(可选,非 P1/P2)**
- 给 `dist` 加运行时重连:邻居检测 dist 连接断开 → 上报 → supervisor 从池租同模型同段 rank 替换 → 重新握手;
- 收益:中间/末 rank 挂时只损失一个层段的 KV(整组重建丢全部 KV),其余 rank 保留。

## 3.6 源码目录结构

```
yllm/
│
├── inference/                    # 推理内核(不感知网络服务/调度, 可独立成库)
│   ├── yllm.h                    # 公共头(类型/内存/张量抽象)
│   ├── platform.c                # 平台抽象(线程/锁/mmap/时间/原子)
│   ├── log.c  log.h              # 日志
│   ├── llf.c  llf.h              # 模型文件格式(.llf 读写)
│   ├── matvec.c matvec.h         # 向量/矩阵内核(标量+SIMD)
│   ├── tokenizer.c               # 分词
│   ├── engine.c engine.h         # 推理引擎(engine_generate, 权重mmap/KV/前向)
│   ├── dist.c  dist.h            # 分布式层流水线(rank间推理协作, 属推理执行)
│   └── convert.c convert_gguf.c convert_safetensors.c  # 模型转换(离线)
│
├── serve/                        # 服务层(网络/调度/生命周期, 依赖 inference)
│   ├── protocol.h                # 帧协议统一定义(rank/server/router 共用)
│   ├── rank.c                    # yllm rank: 常驻推理单元(收帧循环, 调 inference 推理)
│   ├── server.c                  # yllm server: 业务逻辑组(租用rank/队列/会话亲和/SCALE)
│   ├── router.c                  # yllm router: 调度层(注册表/心跳/路由/转发)
│   ├── router_http.c             # OpenAI 兼容 HTTP(JSON/SSE)
│   ├── supervisor.c              # yllm supervisor: 管理节点(机器清单/rank池/租约/拉起/自愈/更新)
│   └── sync.c                    # 文件分发(模型/二进制推送, 日志拉回)
│
├── tools/                        # 命令行工具(独立小工具)
│   └── dump.c                    # llfdump
│
├── main.c                        # 入口: yllm <convert|check|gen|chat|rank|server|router|supervisor|dump>
│
├── docs/                         # 设计文档
├── tests/                        # 单测
└── Makefile                      # 构建: inference 库 + serve 目标
```

### 3.6.1 两块边界与依赖

| 目录            | 包含                                                   | 依赖方向                                | 不含                  |
| ------------- | ---------------------------------------------------- | ----------------------------------- | ------------------- |
| **inference** | 模型文件/引擎/分词/内核/分布式推理/转换                               | 只依赖 C 标准库 + 自身                      | 网络服务、调度、生命周期、服务层协议帧 |
| **serve**     | rank/server/router/supervisor/HTTP/sync + protocol.h | **依赖 inference**(调 engine/dist/llf) | 模型格式、推理算法           |
| **tools**     | dump 等                                               | 依赖 inference                        | —                   |

依赖方向单向:`main.c → serve → inference`。inference 不感知 serve 存在,可独立编译成库单独测试。

### 3.6.2 dist\_worker 迁移(最终移除)

| dist\_worker 现在功能             | 去向                                                             |
| ----------------------------- | -------------------------------------------------------------- |
| `--serve --root`(文件中转)        | → `serve/sync.c`(supervisor 文件分发)                              |
| 控制 worker(run/sync/ping/stop) | → 拆分:`run` 归 `serve/server.c`+`rank.c`;`sync` 归 `serve/sync.c` |
| `--send` 客户端                  | → `serve/router_http.c` 的 OpenAI 面 / `protocol.h` 客户端          |
| spawn/spawn\_gen              | → `serve/supervisor.c`(拉起进程)                                   |

迁移路径:① 保留 dist\_worker.c 作为工作原型,建好 inference/+serve/ 结构;② 按职责拆分到 serve 各文件;
③ 验证 serve 各 cmd 独立可用后删除 dist\_worker.c。

### 3.6.3 帧协议分类

- `inference/dist.c` 内部:rank 间激活帧(X/LOGITS/DONE),推理执行的一部分;
- `serve/protocol.h`:服务层帧(PING/STAT/INFER/DRAIN/REGISTER/HEARTBEAT/SCALE),rank/server/router 共用。

## 4. 代码改造点

| 位置                     | 改动                                                                                |
| ---------------------- | --------------------------------------------------------------------------------- |
| `src/main.c`           | 新增 `cmd_rank` / `cmd_server` / `cmd_router`(复用 gen 的 engine/vocab 初始化),把生成拆成可复用函数 |
| `src/engine.c`         | 抽 `engine_generate(...)`(prompt→tokens→流式 cb),`cmd_gen` 与 `rank` 共用               |
| `src/dist.c`           | `dist_gen` 增加"循环服务"入口或拆分连接建立/推理;常驻复用 dist 连接                                      |
| `src/dist_worker.c`    | 文件服务 `--serve` + sync 逻辑迁往 `cmd_supervisor`;其余迁往 `cmd_rank/server/router`         |
| `src/router_http.c`(新) | OpenAI 兼容 HTTP 层(JSON/SSE 解析与组装)                                                  |
| `Makefile`             | 加 `rank`/`server`/`router` 目标与示例编排                                                |

请求级拆分(单 rank 与多 rank 统一):

```
int engine_generate(Engine* e, Vocab* v, const uint32_t* ids, int nprompt,
                    int ntokens, float temp, float top_p, uint64_t seed,
                    dist_token_cb emit, void* ctx);
// 单 rank: 内部 engine_forward 循环
// 多 rank: 内部按当前 rank 调 dist 收发(等价于现有 dist_gen 主体)
```

## 5. 部署示例(伪命令)

```bash
# 管理节点: supervisor(决策 + 文件分发 + 拉起)
# supervisor 先把模型/二进制 sync 到各机器, 再拉起 rank/server/router

# 机器1(公用 rank 池: qwen3 段 rank + tinyllama 段 rank)
yllm rank --model /models/qwen3.llf --rank 0 --ranks 2 --port-base 9410 \
    --dist-addrs 192.168.1.161,192.168.0.23 --log logs/q3-r0.log
yllm rank --model /models/tinyllama.llf --rank 0 --ranks 1 --port-base 9420 \
    --log logs/t-r0.log
# 机器2
yllm rank --model /models/qwen3.llf --rank 1 --ranks 2 --port-base 9410 \
    --dist-addrs 192.168.1.161,192.168.0.23 --log logs/q3-r1.log

# 业务逻辑组(supervisor 拉起, 注册到所有 router)
yllm server --id server-q1 --model qwen3 --ranks 2 \
    --rank-pool 192.168.1.161:9410,192.168.0.23:9411 \
    --router 127.0.0.1:9400,127.0.0.1:9401 --port 9421 --log logs/server-q1.log
yllm server --id server-t1 --model tinyllama --ranks 1 \
    --rank-pool 192.168.1.161:9420 --router 127.0.0.1:9400,127.0.0.1:9401 \
    --port 9422 --log logs/server-t1.log

# 调度层: 多实例
yllm router --port 9400 --config servers.yaml --log logs/router0.log
yllm router --port 9401 --config servers.yaml --log logs/router1.log

# 客户端: 标准 OpenAI 兼容
curl http://127.0.0.1:9400/v1/chat/completions -d '{
  "model": "qwen3", "messages": [{"role":"user","content":"讲个故事"}],
  "max_tokens": 64, "stream": true}'
```

滚动更新(替换模型版本, supervisor 负责):

```bash
make serve-roll NEW_MODEL=test/v2.llf   # 起新模型 server→就绪→摘旧→DRAIN→停旧, 零中断
```

## 6. 风险与权衡

| 点                   | 说明                                                          |
| ------------------- | ----------------------------------------------------------- |
| rank 串行             | CPU 场景先串行+排队;并发=多线程读同一 mmap,注意 engine 线程安全                  |
| KV 复用               | 目前 dist\_gen 每次请求重建激活缓冲;serve 需持久化 KV cache 以支持 prefix/会话亲和 |
| 一个 server = 一组 rank | router 只需知道 leader 地址,其余 rank 组内常驻连接                        |
| 同机多 server 实例       | 同模型共享 mmap 页缓存省内存;隔离靠进程,线程方案需 engine 线程安全                   |
| 故障检测时效              | 心跳 2s 粒度,检测 dead 到新 server 就绪有冷启动窗口                         |
| 公用池争抢               | 多个 server 同时租用同一批 rank 需 supervisor 分配仲裁(避免过度租用)            |
| router 无状态          | 靠 server 心跳广播自愈注册表;重启的 router 需等下一次心跳补全                     |

## 7. 里程碑

- **P1 最小闭环**:`yllm rank` 常驻(单机单 rank)+ PING/STAT/INFER/DRAIN 帧协议 + 单机直连推理。
  验证:进程启动一次,连续 N 个 INFER 请求不冷启动,结果入日志。跨机多 rank 复用 dist 握手。
- **P2 路由**:`yllm server` 独立进程(租用 rank 组、会话亲和)+ `yllm router` 多实例
  (server 注册表、心跳、模型过滤/least-inflight/kv-aware/prefix-affinity)。
- **P3 生命周期 + OpenAI 面**:supervisor(滚动更新/自愈/扩缩容/机器 agent)+
  router OpenAI 兼容 HTTP(`/v1/chat/completions` + SSE)。

