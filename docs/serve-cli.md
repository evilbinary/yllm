# yllm serve 命令速查

常驻推理服务(supervisor/router/server/rank/hub)的启动、查看、控制与分发。
架构与设计见 `docs/serving-architecture.md`。

> 以下命令均指 yllm 二进制(源码构建后为 `./build/avx2/yllm`,安装后可简写 `yllm`)。
> `make xxx` 只是源码构建时的等价包装(自动带 `--config serve.yaml` 等默认参数)。

## 启动角色进程

```sh
yllm hub --config serve.yaml          # 三合一: supervisor+router+server 同进程, 自动拉起 rank(推荐)
yllm supervisor --config serve.yaml   # 单独 supervisor(管理节点)
yllm router --config serve.yaml       # 单独 router(调度层)
yllm server --config serve.yaml       # 单独 server(业务组)
yllm rank --config serve.yaml         # 单独 rank(推理进程)
```

或用 ctl 拉起(等价,自动带参数):

```sh
yllm ctl start          # 拉起 hub
yllm ctl start r0       # 单独 rank-0
yllm ctl start s0       # 单独 server-0
yllm ctl start rt       # 单独 router
yllm ctl start sv       # 单独 supervisor
```

未指定 `--config` 时默认加载 `serve.yaml`。

## 停止

```sh
yllm ctl stop           # 优雅停止: 所有 rank 发 DRAIN, supervisor 发 QUIT(hub 随之退出)
yllm ctl r0 DRAIN       # 单独下线 rank-0
yllm ctl s0 DRAIN       # 单独下线 server-0
```

## 查看状态

```sh
yllm ctl status
```

输出三段:

```
== 进程列表(本机 yllm 相关) ==
  pid=... build/avx2/yllm hub --config serve.yaml
  pid=... build/avx2/yllm rank ...

== supervisor 节点表 (127.0.0.1:9500) ==      # QUERY_SERVERS 汇总
  server-0   server model=tinyllama    state=ready  addr=127.0.0.1:9420
  rank-0     rank   model=models/...   state=ready
  router-0   router state=ready        addr=127.0.0.1:9400

== rank 状态 (端口基址 9410) ==               # 直连 STAT
  rank-0: OK inflight=0 kv_mb=601.0 uptime_s=...
== server 状态 ==
  server-0: OK inflight=0 ...
== router 状态 ==
  router-0: OK servers=1 inflight=0 ...
```

## 管理命令(ctl)

目标用短别名,避免歧义:

| 别名 | 实际 | 端口 |
|---|---|---|
| `r0` | rank-0 | rank-port-base + 0 |
| `s0` | server-0 | server-port |
| `sv` | supervisor | sv-port |
| `rt` | router | router-port |

```sh
# 位置参数写法(推荐)
yllm ctl r0 PING                    # rank 存活: OK READY
yllm ctl r0 STAT                    # rank 状态: inflight/kv_mb/uptime
yllm ctl r0 QUIT                    # rank 强制退出
yllm ctl sv QUERY_SERVERS           # 节点表快照
yllm ctl sv SCALE 2                 # 扩一组 rank
yllm ctl status                     # 完整状态查看
yllm ctl stop                       # 停止全部
yllm ctl start [r0|s0|rt|sv]        # 拉起角色进程(缺省 hub)

# 选项式写法(等价)
yllm ctl --target r0 --cmd PING
yllm ctl --target sv --cmd SCALE --need-groups 2
```

## 推理请求

```sh
yllm gen --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.llf \
         --vocab models/tinyllama.vocab.txt --prompt "Hello" --tokens 64

# 经 serve 集群(router → server → rank)
yllm router --config serve.yaml --send "tinyllama 64 Hello"
```

`--send` 格式:`"<model> <max_tokens> <prompt>"`,模型名需匹配 serve.yaml 的 `model-name`。
集群返回流式 token,末尾 `DONE <tokens> <ms>`。

## 文件分发(sync)

```sh
# 接收端(目标机器): 起 sync 服务
yllm sync --serve --port 9600 --dir models/

# 发送端(管理机): 推送文件
yllm sync --push local.gguf --to host:9600 --dest models/model.gguf
```

帧协议 `FILE_PUT <path> <size>` + 字节流,自动建目录,含路径穿越防护。

## 排查

```sh
pgrep -af yllm                     # 进程列表
ss -tlnp | grep yllm               # 端口占用(8000/9400/9410/9420/9500)
tail -f logs/hub.out               # hub 日志(节点注册/心跳)
tail -f logs/rank-0.log            # rank 日志
```

## 端口约定(serve.yaml 可改)

| 端口 | 角色 |
|---|---|
| 9500 | supervisor(心跳/QUERY_SERVERS/SCALE) |
| 9400 | router(INFER 入口) |
| 9420 | server(转发) |
| 9410 | rank(推理; 多 rank 递增) |
| 8000 | OpenAI 兼容 HTTP(可关) |

## 源码构建时的 make 包装(等价)

```sh
make hub / make serve / make rank / make server / make router / make supervisor   # 启动
make status / make infer [prompt...] / make ctl ... / make sync-serve / make sync-push
make serve-stop                          # 杀进程(强停)
```
