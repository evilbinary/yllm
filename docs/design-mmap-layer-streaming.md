# yllm 推理引擎：mmap + 层流式加载设计方案

版本：v0.1 ｜ 状态：草案 ｜ 目标平台：Linux 为主（Windows 支持见 §7）

## 1. 背景与目标

### 1.1 问题

- 大模型权重动辄上百 GB（70B × fp16 ≈ 140GB），传统"加载进内存"方案冷启动耗时数十秒，内存峰值高。
- 推理机常同时服务多个模型/多个请求，内存与磁盘带宽是稀缺资源。
- 已有事实：推理时**层是严格顺序访问**的（prefill 逐层 forward），给了"流式"天然时机；且 decode 阶段每 token 复用全部层权重，给了"页缓存复用"天然时机。

### 1.2 方案核心

1. **mmap**：权重文件只读共享内存映射，由内核页缓存按需换入，零拷贝访问权重，不复制到用户堆。
2. **层流式**：按层为单位驱动"预取 → 计算 → 回收"，预取与计算重叠，隐藏磁盘延迟。
3. **自适应常驻**：decode 首 token 后权重已全部驻留页缓存，后续 token 的访问退化为普通指针解引用，**与全量加载性能完全一致**。流式只在冷启动与内存受限时付出代价。

### 1.3 设计目标

| 指标 | 目标 |
|---|---|
| 冷启动首 token 延迟 | 接近 min(总权重/磁盘带宽按需前几层, 全量加载法首 token 时间) |
| 驻留内存 | 可配置预算（如 8GB），由回收策略保证 |
| decode 稳态性能 | 与全量常驻加载一致（页缓存热） |
| 多请求/多进程 | 共享同一份页缓存 |
| 磁盘预取效率 | 层内顺序读，命中顺序带宽 |

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                应用层 (Python / C++ API)                 │
│   Request 调度  ·  KV Cache 管理器（动态内存，不走 mmap） │
├─────────────────────────────────────────────────────────┤
│              Streaming Scheduler（每请求一个）            │
│  状态机: 预取队列 → 就绪层 → 计算 → 回收 / 驻留          │
│  驻留位图 + 预算自适应（§3.3 / §3.6）                    │
├──────────────────┬──────────────────────────────────────┤
│  WeightStore     │  Prefetcher（后台线程 / io_uring）     │
│  mmap 视图       │  策略: WILLNEED 预读 / 逐层 DONTNEED  │
│  mincore 驻留表  │                                       │
├──────────────────┴──────────────────────────────────────┤
│  算子层：CPU (oneDNN/自研)  ·  GPU (cudaMemcpyAsync 双缓冲) │
├─────────────────────────────────────────────────────────┤
│  LLF 权重文件（层主序，页面对齐）                        │
└─────────────────────────────────────────────────────────┘
```

关键原则：

- **只映射一次，视图永驻**：进程启动时对整个文件建立只读共享映射；"加载"的语义完全交给页缓存与预取/回收策略，避免反复 map/unmap 的开销。
- **流式粒度是"层"，不是"张量"**：层是 transformer 中顺序访问的自然单元，且在文件里按层连续排布后，读一层的 I/O 是纯顺序读。
- **prefill 与 decode 走同一套机制**：差异只体现在策略参数上（§4.3）。

## 3. 关键设计

### 3.1 权重文件格式（LLF：Layer-major Layout File）

safetensors/gguf 的布局是"按张量顺序序列化"，无法保证层连续 → 增加转换工具，输出专用层主序格式。

```
LLF v1 文件布局（均按 4K 页对齐；2MB 巨页对齐为可选 build flag）：

┌────────────────────────────────────────┐
│ Header (256B)                          │
│  magic "YLLMLLF1", version             │
│  config: hidden, layers, heads, ...    │
│  n_tensors, n_layers 表                │
├────────────────────────────────────────┤
│ Layer Directory Table (页对齐)          │
│  layer i -> {file_offset, size_bytes}  │  ← 每项均页对齐
│  tensor meta: name/dtype/shape/offset  │
├────────────────────────────────────────┤
│ layer 0: embed                          │
├────────────────────────────────────────┤
│ layer 1..N: transformer block 权重      │
├────────────────────────────────────────┤
│ layer N+1: norm / head                  │
├────────────────────────────────────────┤
│ layer N+2: lm_head（大，特殊优先级）     │
└────────────────────────────────────────┘
```

要点：

- **页对齐与层间 padding**：`offset_i ≡ 0 (mod 4096)`，每层尾部填充至页边界。保证回收某层时不会误伤相邻层的页（否则最后一页共享两层，DONTNEED 会连带逐出邻居）。
- **头部 + 目录独立**：进程只需读前几页即可获得全部元数据，可先于大文件构建张量图。
- **转换工具**：`llm-convert <input.safetensors|gguf> <out.llf>`，并行按层重排并重算对齐；输出后做一次 HEAD 校验与逐层 CRC（可选）。
- **mmap 前的校验**：映射后读目录区即可验证 magic/尺寸，不必全文件读取。

### 3.2 WeightStore（mmap 访问层）

```c
struct WeightStore {
  int   fd;            // O_RDONLY
  void* base;          // mmap(NULL, fsize, PROT_READ, MAP_SHARED, fd, 0)  ← 只读共享
  size_t fsize;
  // 懒映射: 建映射但未读页 → 0 次磁盘 I/O
  const void* layer_ptr(int i)  { return base + dir[i].offset; }
  // 驻留感知
  bool   resident(int i);       // mincore() 逐页检查 → 驻留位图
  // 预取 / 回收（见 §4）
  void   prefetch(int i);       // madvise(MADV_WILLNEED) 或 io_uring 预读
  void   release(int i);        // madvise(MADV_DONTNEED) 仅当策略允许
};
```

- 使用 **MAP_SHARED + PROT_READ**（只读共享映射）：同一模型的多个进程共享同一份页缓存，页错误只在全局首次发生。
- 页错误处理：访问未驻留页时由内核同步 fault（这是流式预取未命中的兜底路径，属预期内的最坏情况）。
- Windows 对应物：`CreateFileMappingW` + `MapViewOfFile`（§7）。

### 3.3 Streaming Scheduler（层流式状态机）

每请求一个实例。状态集合：

```
{ 待预取(P) → 预取中(F) → 就绪(R) → 计算中(C) → 已释放(D) }
预取队列: 后台线程按序拉取
驻留位图: bitset（每层 1 bit）+ 累计驻留字节 + 缺页计数器
```

**驻留位图（residency bitmap）**：mmap 后页缓存驻留状态对程序是黑盒——访问层 i 可能 0 成本命中，也可能同步缺页读盘。位图即"当前 RAM 里真的驻留着哪些层"的账本：bit=1 命中免费，bit=0 访问会缺页。用 `mincore()` 刷新（Windows 无等价 API，退化为"自己 release 自己记"），供预取跳过判断（§4.2）与预算控制器（§3.6）作输入。

prefill 单 token 驱动的主循环（CPU 路径）：

```
for i in 0..last_layer:
    wait_ready(i)                 # 预取完成或同步缺页兜底
    forward(layer i)              # 读映射页，页缓存已热
    if budget_exceeded: release(i-1)   # 回收最旧已算层
    prefetch(i+1)                 # 与 forward(i) 重叠
    adapt_budget()                # 预算自适应（§3.6）
```

- **双缓冲重叠**：预取下一层排在后台线程/io_uring，与当前层计算并发。
- **取消语义**：请求中止时清空预取队列，已提交 readahead 直接丢弃（页缓存留着也无害，可能被后续请求复用）。

### 3.4 GPU 层流式卸载（可选扩展）

模型大于显存时，mmap 页缓存充当"主机侧中间层"，配合双缓冲 DMA：

```
host_buf[0], host_buf[1]   # 页缓存/固定内存
预取层 i+1 到 host_buf[i+1]      ┐ 与
cudaMemcpyAsync(D2D: host→dev) ┘ 层 i 的 kernel 重叠
计算完释放 host_buf[i]
```

约束：流过设备的开销 ≈ 层数 × (页缓存读 + PCIe 拷贝)，适合显存小于模型但大于单层（或几层）的场景；对显存足够场景直接跳过此路径。

### 3.5 KV Cache 与动态张量

KV cache 走普通动态内存（cudaMalloc / 堆），**不 mmap**，原因：随序列长度增长、需要随机读写、回收再分配频繁，且与权重共享页缓存会互相污染。

### 3.6 内存预算与回收策略

- 预算参数 `--weight-resident-budget`（默认 0 = 不限，全驻留）。
- 监控：遍历 `mincore()` 的驻留位图 + 更新计数器，每 N 层或每 1ms 采样一次。
- 超限时按"层号最旧优先"回收（`MADV_DONTNEED`），但**热点层豁免**：
  - `embed` / `lm_head`：每个 token 都访问且 lm_head 可能巨大 → 常驻优先级最高。
  - `norm` 等小层：占用小，恒驻留。
- 回收只是建议性：`MADV_DONTNEED` 后页缓存仍可能被内核重新保留（Linux 语义为"页变为未驻留，内容丢弃"），故回收后**必须标记驻留位图失效**，下次访问按缺页重读。

#### 预算自适应（budget adaptation）

固定预算太死：RAM 富余时浪费性能（层不够驻留→缺页掉速），紧张时回收不过来。自适应 = 带反馈的控制环：**位图提供状态，缺页率与内存压力提供反馈，预算动态浮动**。

```
static int budget = 1;              # 单位: 层数（初始保守）

adapt_budget():                     # 每处理完一层调用
    # 1. 超限回收：驻留最旧层，热点层（embed/head/norm/当前层/下一预取层）豁免
    while resident_layers > budget and has_evictable():
        release(evict_oldest())     # madvise(MADV_DONTNEED) + 清位图 bit
    # 2. 反馈调节（阈值可配）
    pf = page_faults_per_token      # /proc/self/stat (majflt) 或 getrusage
    if pf > FAULT_THRESHOLD and mem_free > FREE_HIGH:
        budget += 1                 # 缺页太多且内存富余 → 多留一层，掉速恢复
    if mem_free < FREE_LOW:
        budget -= 1                 # 内存告急 → 主动缩驻留，防被 OOM 波及
    budget = clamp(budget, 1, max_layers)
```

行为曲线（TinyLlama 1.1B Q4_K_M，层 ≈ 25MB，KV fp16 44MB @2048，运行时 ~50MB 起）：

| 机器可用 RAM | 自适应收敛预算 | 行为 |
|---|---|---|
| ~64MB | 1 层 | decode 每 token 全模型重读，慢速但能跑 |
| ~256MB | ~4-8 层 | 部分层页缓存命中，中等速度 |
| ≥ ~700MB | 全驻留 | 退化为常驻模式，~25 tok/s（内存带宽瓶颈） |

即：**~50MB 可运行，性能随 RAM 平滑伸缩，无需改代码**。RAM 富余时预取深度与预算联动放大（§4.1）。

### 3.7 多架构支持与 MoE 专家级流式

层流式是**架构无关的容器**：所有 decoder-only 模型的差异都落在算子层与 LLF 布局细节，流式框架（mmap/驻留位图/预算/预取）不动。LLF header 增加 `arch` 字段 + 张量名映射表，converter 按架构注册。

| 架构 | 张量名映射（示例） | 差异点（对流式的影响） |
|---|---|---|
| llama | `model.layers.{i}.self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj` | 基线 |
| qwen2 / qwen2.5 | `model.layers.{i}.mlp.{gate,up,down}_proj` 等 | vocab 151K → embed/lm_head 巨大，**常驻豁免优先级提高**（每 token 必读的 100+MB 不容流式重读）；GQA 头数不同只改 KV 公式 |
| qwen3（dense） | 同 qwen2，RoPE θ=1.41M | 只换 RoPE 表，流式无关 |
| qwen3（MoE） | `model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj` + `router` + `shared_expert` | 见下方专家级流式 |

**MoE 专家级流式**：LLF 布局中把 MoE 层的每个专家按页对齐排成独立子块，预取/回收粒度从"层"细化到"专家"：

```
专家子块元数据: expert e → {file_offset, size, 页对齐}
```

- **路由驱动加载**：仅当 router 选中专家 e 才向预取队列提交该子块（235B-A22B 激活 4/256 → 磁盘流量省 ~30 倍）
- **专家 LRU + 共享受访计数**：热专家留在驻留位图（常驻豁免名单动态生成），冷专家用完即 DONTNEED
- gate/attention/shared_expert/router 是小张量 → 恒驻留
- 失效风险：专家权重同层内交错访问会打破顺序预读 → 转换时按"层→专家→张量"排序保证顺序性

## 4. 预取策略

### 4.1 顺序预取（spatial）

- 主路径：`prefetch(i+1)` 在 `forward(i)` 期间执行 → 层间流水线。
- 页粒度：对层 i 的字节区间发 `MADV_WILLNEED`，内核按顺序读拉满磁盘带宽。
- 实现选择（平台相关）：
  - Linux：`madvise(MADV_WILLNEED)`；追求极致再用 `io_uring`/`readahead()` 并发加深队列。
  - 深预取：一次提交后续 2~3 层（`prefetch_depth` 参数），代价是放弃部分内存预算，SSD 下推荐 depth=1~2。

### 4.2 自适应跳过（temporal）

每次访问层 i 前先查驻留位图：

- `resident(i)` 完全驻留 → 跳过预取/回收路径（decode 稳态：每 token 完整体验零 syscall 开销）。
- 部分驻留（如上次被 OOM 波及）→ 只对缺失区间 `WILLNEED`。

这是"流式"与"常驻"两种模式的统一开关：**冷启动走流式，热了自动退化为常驻**。

### 4.3 prefill vs decode 策略对照

| 阶段 | 预取 | 回收 | 期望状态 |
|---|---|---|---|
| prefill（首个 token） | depth=1~2 流水线 | 预算超限时按序回收 | 前台计算不被磁盘阻塞 |
| decode（后续 token） | 全部跳过 | 全部跳过 | 整模型热驻留，性能=常驻 |
| decode + 内存受限 | 预取当层缺页区间 | 按序回收已算层 | 每 token 全层重读，吞吐受磁盘带宽上限 |

## 5. 执行管线的集成

- 算子层读取 `layer_ptr(i)` 返回的指针，与常规张量无差别（零拷贝）。
- prefill：整段序列一次流式过完全部层，多序列请求之间按层错峰（请求 A 算层 i 时，请求 B 预取层 j）。
- 多进程（如按数据并行分片）：共享映射天然去重页缓存；注意多进程同时预取同一层时内核会合并 I/O。
- 掉电/损坏兜底：层加载前可选 CRC 校验（默认关闭，命中 `SIGBUS` 时按文件段重读并告警）。

### 5.1 SIMD 内核与三层派发

热路径是 Q4_K_M 反量化 + matmul，纯 C 下的打法是"intrinsics + 编译期/运行期双层派发"。

**指令集梯队**：

| 指令集 | 位宽 | 每寄存器 fp32 | FMA | 定位 |
|---|---|---|---|---|
| SSE2（x86-64 基线） | 128b | 4 | 无 | 保底 / 非 FMA 平台 |
| SSE4.1 + F16C | 128b | 4 | 无 | fp16→fp32 转换 |
| AVX2 + FMA | 256b | 8 | 有（8 FLOP/核/周期） | **x86 主力** |
| NEON（AArch64） | 128b | 4 | 有（vmlaq，部分核双发） | ARM 主力 |

**反量化四步走（各指令集同一套思路）**：

```
① 装 nibble: 每字节 2 个 4-bit 权重
   AVX2: loadu 后 pshufb 查表拆高低 nibble（2 指令出 32 权重）
   NEON: vld1q_u8 + vshrq_n_u8 + vandq_u8（或 vuzpq 拆半）
   SSE2: and + srli_epi16 拆两趟
② 重建 K-quant block（8 个一组 scale/min 恢复 0-15 → 实际权重）
③ 转 fp32: AVX2 用 F16C（_mm256_cvtph_ps）处理 fp16 输入；NEON 用 vcvt_f32_f16
④ FMA 累加: _mm256_fmadd_ps / vfmaq_f32 → 累加器只在寄存器轮转，永不落内存
```

**纪律**：反量化结果只活在寄存器流水里，每线程 4-8 个累加器 + 2-4 次循环展开，禁现算现存（落一次内存 = 带宽翻倍，decode 吞吐直接腰斩）；禁止 gather/scatter，全走顺序 streaming load。

**三层派发框架**：

```c
// 层 1: 编译期按目标机器定版本
#if defined(__AVX2__) && defined(__FMA__)
  #define KERNEL_AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #define KERNEL_NEON
#else
  #define KERNEL_SSE2
#endif

// 层 2: 运行期 cpuid 派发（GCC/clang: __builtin_cpu_supports；MSVC: __cpuid）
static void (*dequant_mul)(...);        // 函数指针
extern void dequant_mul_sse2(...), dequant_mul_avx2(...), dequant_mul_neon(...);

void simd_init(void) {
    dequant_mul = dequant_mul_sse2;
#ifdef KERNEL_AVX2
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        dequant_mul = dequant_mul_avx2;
#elif defined(KERNEL_NEON)
    dequant_mul = dequant_mul_neon;     // ARM 编译即定
#endif
}

// 层 3: 核内多线程（§线程模型）按行切分，每线程扫连续内存段
```

- 编译两~三份二进制（`-march=x86-64` / `-mavx2 -mfma`），运行时查一次 cpuid，全平台覆盖且冷启动零 syscall
- ARM 双发核显式展开双路 NEON 等效 AVX2 带宽；big.LITTLE 下带宽敏感的解码阶段不往小核分活

**收益预期（8 核桌面，TinyLlama Q4_K_M）**：

| 内核实现 | decode tok/s |
|---|---|
| 纯标量 | 8~12 |
| SSE2 | 14~17 |
| AVX2+FMA（寄存器反量化） | 24~28 |
| AVX-512（如有） | +5~10% |

### 5.2 多线程模型与优化

**两类线程角色，扩容规律完全不同**：

| | 计算线程池 | 预取 I/O 线程（1 个专职） |
|---|---|---|
| 瓶颈 | decode = DRAM 带宽；prefill = FMA 吞吐 | 磁盘队列深度 |
| 加线程效果 | decode 达带宽上限后无增益；prefill 线性扩展 | NVMe 上用 io_uring 深队列（SQE depth 32） |
| 并行单位 | 按输出行/块切分 Q4 反量化+matmul | 批量发 WILLNEED / 预读 |

**线程拓扑（8 物理核桌面）**：

```
核0-5: 计算池（6 核，绑大核）
核6:   预取线程（io_uring 深队列，SCHED_IDLE，绑小核/任意核）
核7:   调度 + 预算控制器（驻留位图更新、budget 调节）
```

**配置要点**：

- **按阶段切核数**：prefill 计算瓶颈 → 全 8 核；decode 带宽瓶颈 → 6 核计算即可拿 ~95% 收益（见下表），运行时 `omp_set_num_threads()` 切换
- **关超线程**：SMT 同核共享 LSU/带宽，memory-bound 下无收益还抢带宽，按物理核数配置
- **核绑定**：`sched_setaffinity` 固定拓扑；big.LITTLE 上计算绑大核、预取/调度丢小核
- **预取背压**：预取进度 = 预取层 − 计算层，超上限（如 3）暂停；驻留位图全 1 时预取线程彻底休眠（零 syscall）

**同步与数据布局（隐蔽杀手）**：

| 错误做法 | 后果 | 对策 |
|---|---|---|
| 层就绪用 mutex/condvar | 每层 2 次加锁，吞吐掉 5-15% | SPSC ring + seq 计数：`while (seq.load() < want) _mm_pause()`，无锁 |
| 驻留位图/层状态多核共享写 | 伪共享，带宽收益磨掉 10-20% | 计数按核分片，bitmap 原子位操作，状态数组 64B cacheline 对齐 |
| 预取线程绑大核 | 抢 decode 带宽 | 绑小核 + SCHED_IDLE |
| per-block 反量化临时缓冲现分配 | malloc 风暴 | 每线程预分配私有缓冲 |

**多序列层面**：请求间层错峰流水线（A 算层 i 时 B 预取层 j）；限定并发序列数 N = 有效带宽 ÷ (638MB × tok/s)，超限排队，避免 thrash；NUMA 双路上权重页 `mbind` 绑定本地 node、核绑对应 socket，防跨 NUMA 拉权重（跨 Socket 带宽仅 ~1/3）。

**线程数 ↔ 性能预期（TinyLlama Q4_K_M 638MB，8 核 DDR4 双通道，AVX2）**：

| 计算线程数 | decode tok/s | 边际增量 | prefill tok/s | 边际增量 |
|---|---|---|---|---|
| 1 | 7~9 | — | 15~20 | — |
| 2 | 13~15 | +6 | 30~40 | +18 |
| 4 | 18~21 | +6 | 60~80 | +35 |
| 6 | 22~24 | +3.5 | 90~120 | +35 |
| 8（物理核） | 24~27 | +2 | 120~160 | +35 |
| 8 + SMT(16 逻辑核) | 24~27 无增益 | ~0 | ~130 略增 | ~+5 |

- decode 边际收益递减极快（1→2 核 +6，6→8 核仅 +2）→ **decode 开 6 计算核，余核派给预取/调度/第二序列**
- prefill 每翻倍核数 ~×1.8 → 全核吃满
- 64MB 磁盘极限模式下 decode 与核数无关（NVMe ≈ 4.7 tok/s），加核只涨占用率不涨吞吐

## 6. 性能指标与验收

| 场景 | 测量项 | 验收线（以 70B fp16 / 40 层 / NVMe 4GB/s 为例） |
|---|---|---|
| 冷启动 prefill | 首 token 延迟 | 流水线稳态下 ≈ 层计算总时间；每层 I/O ~90ms 被隐藏 |
| 冷启动驻留峰值 | 实测 RSS | ≤ 预算（如 8GB），不应等于全模型 |
| decode 稳态 | tokens/s | = 全量加载 baseline ±2% |
| 预取效率 | 页命中率（mincore 采样） | 稳态 ≥ 99%，预取层命中率 ≥ 95% |
| 多进程 | 页缓存去重率 | 第二进程冷启动首 token 显著加速 |

## 7. 平台差异

| 能力 | Linux（主目标） | Windows |
|---|---|---|
| 映射 | `mmap(PROT_READ, MAP_SHARED)` | `CreateFileMappingW` + `MapViewOfFile`，`PAGE_READONLY` |
| 预取 | `madvise(MADV_WILLNEED)` / `readahead` / `io_uring` | `PrefetchVirtualMemory` / `FILE_FLAG_SEQUENTIAL_SCAN`（打开句柄时） |
| 回收 | `madvise(MADV_DONTNEED)` | 无等价 API：依赖内存压力自动换出；提供 `FlushViewOfFile` 仅用于可写映射（本方案不用） |
| 驻留查询 | `mincore()` | `GetProcessWorkingSetSize` / `VirtualQuery` 粒度粗，退化为"整层估算"策略 |

Windows 上"流式回收"能力缺失 → 在 Windows 上默认关闭预算回收，退回"整文件预读 + 页缓存按压力淘汰"模式（等于传统快照加载但省一次拷贝）。

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| 层内随机访问（如 embed 查表是随机页） | 破坏顺序读 | embed/lm_head 常驻豁免；预取降级为按页位图 |
| 页共享导致 DONTNEED 误伤邻居 | 相邻层抖动 | 层间 4K/2MB 对齐 padding（§3.1） |
| SSD 带宽不足（decode 受限时每 token 全读） | 吞吐断崖 | 限流/吞吐目标暴露，提示升级常驻预算 |
| `SIGBUS`（文件被截断/损坏） | 崩溃 | 层访问前可选 CRC；映射区 `PROT_READ` 后总线错误转异常处理 |
| 冷启动抖动（预取未命中） | 首 token 延迟超标 | 同步缺页兜底 + 预取 depth 调大 |

## 9. 落地路线

- **MVP（CPU 推理）**：LLF 格式 + 转换工具 + WeightStore + 顺序预取流水线 + 自适应跳过；验收 §6 前三项。
- **P1**：内存预算回收、mincore 统计采样、io_uring 深预取、GPU 双缓冲 offload。
- **P2**：多进程共享、热点层识别自动常驻、错误恢复（SIGBUS → 重读）。

## 10. 附录：核心接口（伪代码）

```python
# 服务端视角
store  = WeightStore("model.llf", resident_budget=8GB)
engine = StreamEngine(store, prefetch_depth=2, gpu=False)

def generate(req):
    ctx = engine.new_stream()
    # —— prefill（首个 token）——
    for tok in req.prompt_tokens:
        out = ctx.step_prefill(tok)      # 内部: 预取/计算/回收流水线
    # —— decode（稳态热驻留）——
    while not out.eos:
        out = ctx.step_decode()          # 内部: 自适应跳过全部预取/回收
    return ctx.result
```

## 11. 参考

ggml-org/ggml/master/docs/gguf.md