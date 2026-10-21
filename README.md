# Ornith-1.5-122B on Strata

把 **Ornith-1.5-122B**（Qwen3.5-MoE 血统，IQ3_S/IQ3_XXS，50.56 GiB）跑在 **Strata 的 `poly` 分支**上，
单张 **RTX 4080 SUPER 32 GB**（i7-13790F / 95.8 GiB RAM / PCIe 4.0×16）。

| 指标 | 起点 | 现在 | 变化 |
|---|---|---|---|
| **prefill** | 851.3 tok/s | **2306.5 tok/s** | **+171%** |
| **8K TTFT** | 4.66 s | **2.079 s** | **−55%** |
| **decode**（安全档） | 47.6 tok/s | **51.4 tok/s** | +1~7% |
| **decode**（turbo 有损档） | — | **54.8~56.2 tok/s** | **+15~18%** |
| 加载 | 12.1 s | 9.9 s | −18% |

## 收益从哪来（三块，**全部零改码**）

| # | 改动 | 收益 |
|---|---|---|
| 1 | `STRATA_PIN_ARENA_GIB=44`（48 层全 pin，43.03 GiB） | **prefill +55.4%**（1028 → 1597.7） |
| 2 | `--prompt-batch 4800` + `--prompt-cache-every 0`（4,794 token 一次吞完） | **prefill +41.8%**、TTFT −29.5% |
| 3 | `--window 4 --lookup 3`（MTP 自适应，**绝不加 `STRATA_DET_WINDOW`**） | **decode +11.4%**（把 #2 吃掉的 decode 补回来） |

组合后：**prefill 851.3 → 2306.5（+171%）**、**TTFT 4.66 → 2.079 s（−55%）**。

## 仓库内容

```
src/poly/engine.hpp      改动后完整源码（T8 双缓冲字段 + 第 17 落点）
src/poly/engine.cpp      改动后完整源码（T8 + 落点修复）
src/poly/main.cpp        改动后完整源码（T8 入口/出口 + NaN 守卫 + 注入钩子）
serve/server.py          改动后完整源码（eos 248044）
patches/                 四个相对上游的 unified diff（共 340 行）
configs/                 安全档 / turbo 有损档 的 run config
benchmark/performance.md 完整读数表、判否清单（20+ 条，带判据）、方法论
```

## 三个补丁的结局（诚实记录）

| 补丁 | 行数 | 结局 |
|---|---|---|
| **T8 跨层双缓冲** | 17 落点 / 2 次编译 | ❌ **负收益 −18.5%**（机制跑通：不抛异常、L3 行正常、staging 1×、`hold reuses` 恒 0；但 chunk 放大让 4,794 token 变 **1-call** ⇒ "下一层还没搬"这个前提消失 ⇒ 47 次预取纯多付 DMA） |
| **NaN 守卫** | +65 行（`main.cpp`） | ✅ **通过**：不设 env 时 4 条输出与基线**逐字节同**；`STRATA_NAN_INJECT=1` 时出 `nan guard: … dropping KV + rerun` 且输出不再是 `<unk>` |
| **eos 248044** | 1 行（`server.py`） | ✅ **通过**：模型 `config.json` 写 `eos_token_id: 248044`，而 GGUF 只声明 248046 ⇒ serve 只认一个；补上后启动自证变 `end-of-reply ids: 248044,248046`，4 条输出 sha256 仍全同（位精确） |

## 两个"看起来像智力退化、其实是配置"的坑

1. **`max_tokens` 被 thinking 吃光**：`usage` 里 **thinking 计入 `completion`**，且**没有 `reasoning_tokens` 字段**。开思考时 `max_tokens` 给小了（如 1024）会出现 **content 0 字 + `finish=length`** —— 看起来像"模型不会答"，实际是预算不够。**修法：`max_tokens` ≥ 4096**。
2. **"身份混乱"是模型权重，不是引擎**：身份题问三次，中文稳定答「我是 Claude，由 Anthropic 开发」、英文 2/3 答「Zhipu AI / GLM-4.5」—— 这是**微调数据里的身份污染**，接线侧（tokenizer / template / eos / `<unk>`）全部验过无误。**判身份只能看一致性，不能看"说了哪个名字"**（`qwen3_5_moe` 的基座本来就是 Qwen3.5，"通义千问"是它的真名）。

## 环境与复现

```powershell
# 1) 引擎（CUDA 13.3 编译，DLL 在 bin\x64 不在 bin）
$env:PATH = "<cuda-13.3>\bin\x64;$env:PATH"

# 2) 安全档（默认）
$cfg = configs/ornith122b-serve.json
# args: --prompt-batch 4800 --prompt-cache-every 0 --window 4 --lookup 3
#       --pcie-frac 0 --stage-mib 2048 --threads 16 --max-context 32768
# env : STRATA_PIN_ARENA_GIB=44  STRATA_STAGE_RING2=0  STRATA_MAX_PROMPT_BATCH=8192
#       STRATA_PROFILE_DECODE_ONLY=1  STRATA_AHEAD_EXACT=1  STRATA_SCRATCH_TILE=1

# 3) 位置：`STRATA_MAX_PROMPT_BATCH` 必须 8192（4096 = 空转，源码只在 >4096 时抬上限）
# 4) 禁：STRATA_DET_WINDOW（强制满仓投机，会把 +6% 变成 −24%）
```

**turbo 档**（有损）：在安全档基础上加 `STRATA_ROUTE_TAIL_SKIP=7` —— decode **+18%**，代价是**每 token 跳过约 14.2 次专家使用**、输出会变。**默认不要开**。

## 数据口径（重要）

- **prefill 只认 `reused=0`**：引擎每 4,096 token 存检查点，只改 prompt 末尾会走"4096 前缀 + 尾"混合路径 ⇒ 报出 2 倍假高。
- **decode 必须用长答案**：5~7 token 的短答案会把固定开销摊在极短窗口上 ⇒ 系统性低估 30~40%。
- **判否必须带条件**：同一个 `--prompt-batch 4096` 在旧条件下是 −3.03%，条件全变后（新 exe + pin + RING2=0）变成 **+41.8%** —— 条件变了，判否就失效。
