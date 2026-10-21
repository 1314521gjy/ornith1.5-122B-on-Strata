# Ornith-1.5-122B on Strata

把 **Ornith-1.5-122B**（Qwen3.5-MoE 血统，IQ3_S/IQ3_XXS，50.56 GiB）跑在 **Strata 的 `poly` 分支**上，
单张 **RTX 4080 SUPER 32 GB**（i7-13790F / 95.8 GiB RAM / PCIe 4.0×16）。

| 指标 | 起点 | 现在 | 变化 |
|---|---|---|---|
| **prefill** | 851.3 tok/s | **2306.5 tok/s** | **+171%** |
| **8K TTFT** | 4.66 s | **2.079 s** | **−55%** |
| **decode**（安全档） | 47.6 tok/s | **48.7–51.4 tok/s** | +1~7% |
| **decode**（turbo 有损档） | — | **54.8–56.2 tok/s** | **+15~18%** |
| 加载 | 12.1 s | 9.9 s | −18% |

## 🚀 想直接部署？看 [DEPLOY.md](DEPLOY.md)

那里有完整的闭环：**模型获取与 sha256 校验 → 编译 → 一键启动脚本 → 健康检查期望值 → 踩过的 10 个坑**。
本仓库自带 **tokenizer**（从 GGUF 导出，不要另取）与 **专家画像表**（10,425 行），拿到就能跑。

```powershell
# 最短路径
.\scripts\start-safe.ps1 -Model <第一片.gguf> -Exe <strata-poly.exe> `
                         -Profile .\profile\expert-profile-ornith122b.txt -Tokenizer .\tokenizer
```

## 📦 预编译引擎（可选）

`bin/strata-poly.exe` —— **62.62 MB** / sha256 `03885B2402A4CAB3…` / Windows x64 + CUDA 13.3 + sm_89。

- **DLL 不随仓库分发**（`cublasLt64_13.dll` 有 438 MB），装 **CUDA 13.3 Runtime** 即可；DLL 在 `<cuda>\bin\x64`，**不在** `<cuda>\bin`
- **别人的机器上更推荐自己编**（76 秒，16 线程），见 [`bin/README.md`](bin/README.md) 与 [`DEPLOY.md`](DEPLOY.md) §3
- 硬要求：**显存 ≥32 GB**（24 GB 跑不了）、RAM ≥64 GiB、sm_89

## 收益从哪来（三块，**全部零改码**）

| # | 改动 | 收益 |
|---|---|---|
| 1 | `STRATA_PIN_ARENA_GIB=44`（48 层全 pin，43.03 GiB） | **prefill +55.4%**（1028 → 1597.7） |
| 2 | `--prompt-batch 4800` + `--prompt-cache-every 0`（4,794 token 一次吞完） | **prefill +41.8%**、TTFT −29.5% |
| 3 | `--window 4 --lookup 3`（MTP 自适应，**绝不加 `STRATA_DET_WINDOW`**） | **decode +11.4%**（把 #2 吃掉的 decode 补回来） |

组合后：**prefill 851.3 → 2306.5（+171%）**、**TTFT 4.66 → 2.079 s（−55%）**。

## 仓库内容

```
README.md                本件
DEPLOY.md                ★ 部署指南（模型/编译/启动/健康检查/坑）
src/poly/engine.hpp      改动后完整源码（T8 双缓冲字段 + 第 17 落点）
src/poly/engine.cpp      改动后完整源码
src/poly/main.cpp        改动后完整源码（T8 + NaN 守卫 + 注入钩子）
serve/server.py          改动后完整源码（eos 248044）
patches/                 四个相对上游的 unified diff（共 340 行）
configs/                 安全档 / turbo 有损档 的 run config
scripts/                 start-safe.ps1 / start-turbo.ps1（一键启动）
profile/                 专家画像表（10,425 行，103,504 B）
tokenizer/               tokenizer（从 GGUF 导出；serve 必需）
benchmark/performance.md 性能读数 + 20+ 条判否（带判据）+ 三条方法论 + 五条口径坑
benchmark/intel-test.md  智力测试（28 题 / 32 请求）逐题明细
```

## 智力测试（验"有没有退化"）

| 维度 | 结果 |
|---|---|
| **算数 15**（5×3 长度难度矩阵） | **15/15 全对**，5 道干扰题全部没上当 |
| **对话 12** | 技术通过（无 `<unk>`、无空答案、`reasoning_len` 全 >0） |
| **工具 4** | 3/4 触发 native `tool_calls`、args 合法；第 4 条是"有副作用工具先要授权"的安全行为 |
| **代码-SVG 1** | 四条全过（合法 XML、闭合、有 `@keyframes`、`finish=stop`） |

**⇒ 该仪器集上未见智力退化**（28 题不够做统计判据；本机《质量闸门》要求 ≥400 题才能判小退化）。
明细见 [benchmark/intel-test.md](benchmark/intel-test.md)。

**唯一硬伤：身份污染。** 中文身份题 **3/3 答「我是 Claude，由 Anthropic 开发」**、英文 **2/3 答「Zhipu AI / GLM-4.5」**、1 次回避，且**跨会话不一致** —— 这是**模型权重里的训练数据污染**（接线侧 tokenizer/template/eos/`<unk>` 全部验过无误），**要重训或换权重才能修**。

## 三个补丁的结局（诚实记录）

| 补丁 | 行数 | 结局 |
|---|---|---|
| **T8 跨层双缓冲** | 17 落点 / 2 次编译 | ❌ **负收益 −18.5%**（机制跑通：不抛异常、L3 行正常、staging 1×、`hold reuses` 恒 0；但 chunk 放大让 4,794 token 变 **1-call** ⇒ "下一层还没搬"这个前提消失 ⇒ 47 次预取纯多付 DMA） |
| **NaN 守卫** | +65 行（`main.cpp`） | ✅ **通过**：不设 env 时 4 条输出与基线**逐字节同**；`STRATA_NAN_INJECT=1` 时出 `nan guard: … dropping KV + rerun` 且输出不再是 `<unk>` |
| **eos 248044** | 1 行（`server.py`） | ✅ **通过**：模型 `config.json` 写 `eos_token_id: 248044`，而 GGUF 只声明 248046 ⇒ serve 只认一个；补上后启动自证变 `end-of-reply ids: 248044,248046`，4 条输出 sha256 仍全同（位精确） |

## 两个"看起来像智力退化、其实是配置"的坑

1. **`max_tokens` 被 thinking 吃光**：`usage` 里 **thinking 计入 `completion`**、**没有 `reasoning_tokens` 字段**。开思考时给 1024 会出现 **content 0 字 + `finish=length`** —— 看起来像"模型不会答"，实际是预算不够。**修法：`max_tokens` ≥ 4096。**
2. **"身份混乱"是模型权重，不是引擎**（见上）。

## 数据口径（重要）

- **prefill 只认 `reused=0`**：引擎每 4,096 token 存检查点，只改 prompt 末尾会走"4096 前缀 + 尾"混合路径 ⇒ 报出 2 倍假高。
- **decode 必须用长答案**：5~7 token 的短答案会把固定开销摊在极短窗口上 ⇒ 系统性低估 30~40%。
- **判否必须带条件**：同一个 `--prompt-batch 4096` 在旧条件下是 −3.03%，条件全变后（新 exe + pin + RING2=0）变成 **+41.8%** —— 条件变了，判否就失效。

## ⚠️ 已知问题（接入真实应用后才暴露的）

| 问题 | 表现 | 真相 |
|---|---|---|
| **`thinking` 无上限** | 一问要等几分钟 | **`max_tokens` 是 thinking 的预算**（`usage` 里 thinking 计入 `completion`、无 `reasoning_tokens` 字段）。给 65536 会让它思考 **30 分钟**；**建议 4096–8192**。给 1024 会出现 content 0 字 + `finish=length` |
| **`--prompt-cache-every 0` 让多轮变慢** | 每问一句都全量 prefill 30K（**18 s**） | 那是**单轮** +41.8% 的代价。**多轮/agent 场景必须用 `2048`**（第 2 轮起 **0.38 s**） |
| **长上下文明显慢** | 128K 时 TTFT **128 s**、decode **19.5**（8K 时 2.11 s / 45.3） | KV **按需增长** ⇒ **不挤专家缓存（5,544 未缩）、不 OOM**，只是 attention 本身贵 |
| **身份污染** | 问"你是谁"答 Anthropic / GLM | 模型权重里的训练数据污染，**修不了**；**不影响技术能力**（算数 15/15、工具 3/4、SVG 四条全过） |

详见 [`benchmark/multiturn-and-context.md`](benchmark/multiturn-and-context.md)。
