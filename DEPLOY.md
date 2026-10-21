# 部署指南（Deploy）

> 目标：**拿到这个仓库就能把这套跑起来。**
> 本仓库验证环境：**RTX 4080 SUPER 32 GB** · i7-13790F（16 物理核）· RAM 95.8 GiB · PCIe 4.0×16 · Windows 11

---

## 0 · 你需要什么

| 项 | 要求 | 备注 |
|---|---|---|
| **GPU** | ≥ 32 GB 显存 | 实测峰值 **31,645 MiB**（安全档）。24 GB 卡**跑不了**（专家缓存会挤到 0） |
| **RAM** | ≥ 64 GiB | 实测 pin 43.03 GiB（锁页）+ 模型 mmap 约 50 GiB；**建议 96 GiB** |
| **磁盘** | ≥ 60 GiB 空闲 | 模型 50.56 GiB + 构建产物 |
| **CUDA** | **13.3** | DLL 在 `<cuda>\bin\x64`（**不在 `bin`**，这条踩过） |
| **构建** | CMake 4.x + MSVC 14.51 + Ninja | 见 §3 |

---

## 1 · 模型

**本仓库验证的是**：`Ornith-1.5-122B-AD-IQ3_S-IQ3_XXS`（第三方 IQ3 量化，**两片 GGUF**）

| 文件 | 大小 | sha256（前 16） |
|---|---|---|
| `Ornith-1.5-122B-AD-IQ3_S-IQ3_XXS-00001-of-00002.gguf` | 44,741,990,560 B | `8F485EF8…F8E8C5` |
| `Ornith-1.5-122B-AD-IQ3_S-IQ3_XXS-00002-of-00002.gguf` | 9,544,861,792 B | `55C95BF2…F95E1A` |
| **合计** | **54,286,852,352 B（50.56 GiB）** | |

- **模型不上传**（50 GiB）。请自行从 HuggingFace 取 **Ornith-1.5-122B 的 IQ3_S/IQ3_XXS 量化**，**并逐片校验 sha256**。
- 两个分片放**同一目录**，文件名保持 `-00001-of-00002` 格式（引擎按此规则找第二片）。
- **tokenizer 已随仓库附带**（`tokenizer/`，从 GGUF 导出），**不要再从 HF 另取** —— 版本不同会导致 chat template 不一致。

**模型几何（校验用）**：
```
arch = qwen35moe        layers = 48 (+1 MTP)   experts = 256, top-8
hidden = 3072           moe_intermediate = 1024  shared = 1024
q heads 32 / kv 2 / head_dim 256                GDN : full attention = 3 : 1 (36 + 12)
ssm.state_size 128      rope.dimension_count 64  sections [11,11,10]  theta 1e7
vocab 248,320           eos 248,046             max_position 262,144
```

---

## 2 · 引擎

本仓库的 `src/poly/` 与 `serve/server.py` 是 **改动后的完整文件**，基线是
**Strata `v0.1.44.1-poly.3`**（commit `eb5ef118`，2026-10-15）。

| 文件 | 改动 | 结局 |
|---|---|---|
| `src/poly/engine.hpp` / `engine.cpp` | T8 跨层双缓冲（17 落点）+ `prefetched_il_ < il` 修复 | ❌ **负收益**（chunk 放大后无对象），`RING2=0` 默认关 |
| `src/poly/main.cpp` | 同上 + **NaN 守卫**（+65 行，含 `STRATA_NAN_INJECT` 钩子） | ✅ 通过（不设 env 逐字节同） |
| `serve/server.py` | **eos 补 248044**（1 行，`eos_ids_of()`） | ✅ 通过（位精确，启动自证两个 eos） |

**两条路**：
- **A（推荐，最省事）**：直接用本仓库的文件覆盖到你的 PolyStrata 树（同名路径），然后按 §3 编译。
- **B（最小改动）**：只用 `patches/` 里的 4 个 diff（共 340 行）打到你的树 —— 其中**只有 NaN 守卫与 eos 值得打**（T8 是负收益，可跳过）。

---

## 3 · 编译

```powershell
# 1) CUDA 环境（DLL 在 bin\x64，不在 bin —— 缺了会 exit 0xC0000135 且日志一字不写）
$env:PATH = "<your-cuda-13.3>\bin\x64;$env:PATH"

# 2) 构建（改了 engine.hpp ⇒ 全部 10 个 cpp 重编 + 链接，约 76 s @ 16 线程）
cmake --build "<PolyStrata>\build-p3" --target strata-poly -j 16
```

**编译成功的三条判据**（缺一条就是假绿）：
1. `error` 0 条
2. `strata-poly.exe` 的 **mtime 晚于 `engine.cpp`**
3. 日志里既有 `Building CXX object ...main.cpp.obj`（或 engine.cpp），也有 `Linking CXX executable strata-poly.exe`

---

## 4 · 启动

```powershell
$env:PATH = "<your-cuda-13.3>\bin\x64;$env:PATH"

.\scripts\start-safe.ps1 `
  -Model   "D:\models\Ornith-1.5-122B-AD-IQ3_S-IQ3_XXS-00001-of-00002.gguf" `
  -Exe     "D:\build-p3\strata-poly.exe" `
  -Profile ".\profile\expert-profile-ornith122b.txt" `
  -Tokenizer ".\tokenizer"
```

（`scripts/start-turbo.ps1` 是同一条命令 + `STRATA_ROUTE_TAIL_SKIP=7`，**有损**，见 §6）

**启动后必须看到的自证行**（没有就是没生效）：
```
pinned 48 layers (43.03 GiB) of 49, 0 refused          ← pin 生效
prompt batch: 4800 tokens a call (918 MiB staging)     ← chunk 生效
[strata] end-of-reply ids: 248044,248046               ← 两个 eos 生效
```

**第一条请求必须验 `reused=0`** —— 引擎每 4,096 token 存检查点，若 prompt 前缀与历史相同会走"4096 前缀 + 尾"混合路径，**prefill 会虚高 2 倍**。

---

## 5 · 健康检查（跑一次就知道有没有接对）

```powershell
# 8K prompt + 长答案；看 TTFT / prefill / decode
python ..\_tools\bench_prefill.py --base-url http://127.0.0.1:8123 ...
```

**期望值（RTX 4080S 32 GB）**：

| 指标 | 安全档 | 说明 |
|---|---|---|
| **prefill** | **2306.5 tok/s** | `reused=0` 口径 |
| **8K TTFT** | **2.079 s** | |
| **decode** | **48.7–51.4 tok/s** | 长答案（150–250 字）口径；短答案会**系统性低估 30–40%** |
| 专家缓存槽 | **5,544** | |
| 峰值显存 | **31,645 MiB** | >32,600 就要停下来查 |

**偏差 >15% 就是没接对**，按顺序查：① `pinned` 行在不在 ② `prompt batch` 是不是 4800 ③ pin 的 GiB 数对不对（43.03）④ `STRATA_MAX_PROMPT_BATCH` 是不是 **8192**（设 4096 = 空转，源码只在 >4096 时抬上限）。

---

## 6 · 两档的区别

| | 安全档（默认） | turbo 档 |
|---|---|---|
| config | `configs/ornith122b-serve.json` | `configs/ornith122b-serve-turbo.json` |
| 额外 env | — | `STRATA_ROUTE_TAIL_SKIP=7` |
| decode | 48.7–51.4 | **54.8–56.2（+18%）** |
| 代价 | — | **每 token 跳过约 14.2 次专家使用**、输出会变（上游测得 KL≈0.0058） |
| 质量 | 算数 15/15、SVG 四条全过 | **未测** |

⚠️ **turbo 档是"用质量换速度"，默认不要开。** 任何对外报告引用 56 tok/s 时必须同时标注"有损、每 token 跳 14.2 次专家"。

---

## 7 · 踩过的坑（省你几天）

| 坑 | 表现 | 正解 |
|---|---|---|
| **CUDA DLL 在 `bin\x64`** | `exit 0xC0000135`，日志**一字不写** | PATH 指 `bin\x64` |
| **`--serve` 不是 HTTP** | 起来但连不上 | 它是 stdin/stdout 行协议；HTTP 层在 `serve\server.py` |
| **env 要 `STRATA_` 前缀** | 设了没反应 | 裸名不生效 |
| **`STRATA_MAX_PROMPT_BATCH=4096`** | 设了等于没设 | 只在 >4096 时抬上限 ⇒ 要 **8192** |
| **`STRATA_DET_WINDOW=1`** | 投机从 +6% 变 −24% | 那是"强制满仓"，永远别加 |
| **短答案测 decode** | 低估 30–40% | 必须 150–250 字长答案 |
| **`reused=4096`** | prefill 虚高 2 倍 | 只认 `reused=0`，唯一标记放 prompt **最前面** |
| **`max_tokens` 被 thinking 吃光** | content 0 字 + `finish=length` | `usage` 里 **thinking 计入 `completion`**、无 `reasoning_tokens` 字段 ⇒ 开思考时 `max_tokens` ≥ 4096 |
| **`--expert-swaps` 默认 64** | 引擎 run-to-run 不可复现 | 要位精确必须 `--expert-swaps 0 --lookup 0` |
| **git 全局有 URL 特定代理** | push 卡在 `127.0.0.1:7897` | `git -c "http.https://github.com/.proxy=" push` |
