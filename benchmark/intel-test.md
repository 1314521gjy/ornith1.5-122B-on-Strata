# Ornith-1.5-122B on PolyStrata · 智力测试报告（2026-10-21）

> **一句话**：在 28 题 / 32 请求的仪器集上，**未见智力退化** —— 算数 **15/15**、工具 **3/4**（第 4 条是有副作用工具，模型先拟稿再要求确认）、SVG **4/4 通过**、对话 **12/12 干净**（无 `<unk>`、无空答案、无预算截断）；**唯一硬伤是身份认知**（中文 3/3 自称 Claude/Anthropic、英文 2/3 自称 Zhipu/GLM-4.5），属**微调权重问题，不是引擎接线问题**。

## 1. 环境与口径

| 项 | 值 |
|---|---|
| 模型 | Ornith-1.5-122B（`qwen35moe`，IQ3_S/IQ3_XXS 两片） |
| 引擎 | PolyStrata `strata-poly`（exe sha256 `03885B24…E26B`，65,662,464 B） |
| 服务层 | `serve/server.py`（sha256 `DE3F98BD…5C33`，447,518 B） |
| 采样 | `temperature: 0`、`reasoning_effort: "high"`（**开思考**） |
| 输出预算 | 对话 / 算数 / 工具 **4096**，SVG **8192**（`max_tokens`） |
| 安全档 | `--tail-skip 0`（无有损项）、`--pcie-frac 0`、`STRATA_PIN_ARENA_GIB=44` |
| 题集 | `_raw/intel-test-v1.jsonl`（28 题：chat 8 / arith 15 / tool 4 / svg 1） |
| 请求数 | **32** = 6 普通对话 + 2 身份题 ×3 重复 + 15 算数 + 4 工具 + 1 SVG |
| 判定方式 | 题集 `expect` 字段给期望类型（`final_number` / `must_call:<tool>` / `follows_instruction` / `on_topic` / `no_repeat` / `same_identity_or_avoid` / `svg|anim`）；题集 `scored` 未填 ⇒ **对错由本轮人工按 `expect` 逐条判** |

## 2. 结果总表

| 维度 | 题量 / 请求 | 结果 | 判定 |
|---|---|---|---|
| **对话** | 12 条（6 普通 + 2 身份×3） | 无 `<unk>`、无空答案、无 `finish=length`；`reasoning_len` 全 > 0（53–4,215） | ✅ 技术通过 |
| **算数** | 15（5×3 难度矩阵；中 13 / 英 2） | **15/15 全对**；easy 5/5、med 5/5、hard 5/5；**5 道带干扰项的题全部没上当** | ✅ **最强证据** |
| **工具** | 4 | **3/4 触发 native `tool_calls`**（args JSON 合法、required 齐全）；`send_email` 未触发（先拟稿 + 要授权） | ✅ 技术 3/4；第 4 条按**有副作用工具**的安全判据算通过 |
| **代码-SVG** | 1（两档预算） | 4096 档 `finish=length`（预算截断）；**8192 档 `finish=stop`、4,492 tok、SVG 子树 5,883 字符、`@keyframes` 在、XML 合法** | ✅ 通过 |

## 3. 对话 12 条（逐条）

| # | label | 身份题 | 重复 | finish | completion tok | reasoning_len | 正文长（字） | 一句话读数 |
|---|---|---|---|---|---|---|---|---|
| 1 | chat-zh-1 | 否 | 1 | stop | 51 | 53 | 53 | 「我是 Ornith Team 开发的 Ornith Team，一个乐于助人…的助手」 —— 自报身份标签，非真实厂牌 |
| 2 | chat-zh-2-id | **是** | 1/3 | stop | 96 | 106 | 94 | 「我是 Claude，由 Anthropic 开发」 |
| 3 | chat-zh-2-id | **是** | 2/3 | stop | 184 | 283 | 89 | 「我是 Claude，由 Anthropic 开发；版本不详」 |
| 4 | chat-zh-2-id | **是** | 3/3 | stop | 189 | 244 | 136 | 「我是 Claude，由 Anthropic 开发」 |
| 5 | chat-en-3-id | **是** | 1/3 | stop | 75 | 182 | 71 | 「Zhipu AI, based on GLM-4.5」 |
| 6 | chat-en-3-id | **是** | 2/3 | stop | 88 | 227 | 91 | 「Zhipu AI, based on GLM-4.5」 |
| 7 | chat-en-3-id | **是** | 3/3 | stop | 135 | 209 | 397 | 回避：「I don't have a specific model name」 |
| 8 | chat-zh-4 | 否 | 1 | stop | 840 | 1,071 | 50 | 三句话讲蓝天 + 末行「完」 —— **格式与字数都守住了** |
| 9 | chat-en-5 | 否 | 1 | stop | 135 | 379 | 168 | 三条编号列举（1-3） |
| 10 | chat-zh-6 | 否 | 1 | stop | **1,277** | **3,681** | **643** | 杭州出差三分类建议（含"我看不到实时预报，今晚再核对"的诚实边界） |
| 11 | chat-en-7 | 否 | 1 | stop | 1,291 | 4,215 | 1,490 | 广州→北京一月面试的打包/行程建议 |
| 12 | chat-en-8 | 否 | 1 | stop | 118 | 323 | 168 | 商务邮件礼貌改写的开头 |

> 身份题读数：**中文 3/3 稳定「Claude / Anthropic」；英文 2/3「Zhipu AI / GLM-4.5」+ 1 次回避**；且**跨会话不一致**（另一轮曾答"通义千问"）。⇒ 权重里的身份污染，**不是接线**（tokenizer 单 token 全绿 / chat template 单轮逐字节对 / eos 两个都认 / `<unk>`=0，全部单独验过）。

## 4. 算数 15 条（逐题判定）

| # | label | 题（一句） | 模型答 | 期望 | 判 | 干扰项 |
|---|---|---|---|---|---|---|
| 1 | arith-easy-1 | 12 个苹果 + 5 | **17** | 17 | ✅ | — |
| 2 | arith-easy-2 | 8 支/盒 × 6 盒 | **48** | 48 | ✅ | — |
| 3 | arith-easy-3 | $15 × 4 | **60** | 60 | ✅ | — |
| 4 | arith-easy-4 | 36 辆 − 9 辆 | **27** | 27 | ✅ | — |
| 5 | arith-easy-5 | 7 排 × 9 把 | **63** | 63 | ✅ | — |
| 6 | arith-med-1 | 200 箱 − 45 − 2×45 | **65** | 65 | ✅ | — |
| 7 | arith-med-2 | 40×3/5 中的 1/4 | **6** | 6 | ✅ | — |
| 8 | arith-med-3 | 50 L − 350 km×8 L/100 km | **22** | 22 | ✅ | — |
| 9 | arith-med-4 | 套餐 4 支 10 元 vs 单支 3 元，12 支 | **6** | 6 | ✅ | — |
| 10 | arith-med-5 | 12×8 → 15×6 的面积 | **90** | 90 | ✅ | — |
| 11 | arith-hard-1 | 2400 m：¼ + ⅜×¼ + 420 后剩多少 | **1155** | 1155 | ✅ | 下雨停工 / 调走工人（**明说未改变总量**） |
| 12 | arith-hard-2 | 45 kg 与 30 kg 调平，倒多少 | **7.5** | 7.5 | ✅ | 当天气温湿度（**无关**） |
| 13 | arith-hard-3 | 60 件进 80 卖 40@120 + 20@60 的毛利 | **1200** | 1200 | ✅ | 300 元水电费（**题目明说不计**） |
| 14 | arith-hard-4 | 390 km 两段（240@80 + 150@50）平均速度 | **65** | 65 | ✅ | 两次乘客停站（**不计入给定行驶时间**） |
| 15 | arith-hard-5 | 甲 6 h、乙 12 h 同开，多久注满 | **4** | 4 | ✅ | 排水口（**明说关闭**）/ 无关温度记录 |

**15/15 全对**，其中 **5 道带干扰项的题（#11–#15）全部无视干扰、只取有效条件** —— 这是本报告最强的能力证据（比"答对简单题"强得多）。

## 5. 工具 4 条（native `tool_calls`）

| # | label | finish | tool | args |
|---|---|---|---|---|
| 1 | tool-1-verbose-weather | `tool_calls` | `get_weather` | `{"city":"杭州","date":"tomorrow","unit":"celsius"}` |
| 2 | tool-2-verbose-fx | `tool_calls` | `convert_currency` | `{"amount":250,"from":"USD","to":"CNY","date":"today"}` |
| 3 | tool-3-minimal-read | `tool_calls` | `read_file` | `{"path":"C:\\work\\notes.txt"}` |
| 4 | tool-4-minimal-mail | `stop`（未调用） | — | 输出邮件草稿：收件人 `boss@example.com` / 主题「周报」/ 正文一句话 |

**判据分开写**：
- 按题集原期望（`must_call:send_email`）⇒ **3/4**；
- 按"**有副作用的工具应当先确认**"的安全判据 ⇒ **4/4**（模型主动拟稿并请示，没有擅自发信）。
两条都记，别混成一句"4/4 通过"。

## 6. 代码-SVG（两档预算）

| 档 | finish | completion tok | 正文长 | `<svg>…</svg>` | SVG 子树（字符） | `@keyframes` | 判定 |
|---|---|---|---|---|---|---|---|
| 4096 | **length** | 4,096 | 7,983 | 有 | — | 有 | **预算截断**（非能力问题） |
| **8192** | **stop** | **4,492** | 8,848 | 有 | **5,883** | **有** | ✅ **4/4 全过** |

产物：`_raw/intel-b1-svg-only.svg`（6,174 B，纯 SVG 子树）、`_raw/intel-b1-svg-8192.html`（9,746 B，含 HTML 外壳）、原始 JSON：`_raw/intel-b1-svg.json` / `_raw/intel-b1-svg-8192.json`。

## 7. 两个"假退化"（本轮最重要的排除项）

1. **`max_tokens` 被 thinking 吃光，看起来像"不会答"**：`usage` 里**没有 `reasoning_tokens` 字段**，thinking 计入 `completion`。1024 预算时 `chat-zh-6` 是 **content 0 字 + `finish=length`**（1024 档原始未入档 ⇒ **未验**）；提到 4096 后 **1,277 tok / 643 字 / `stop`**（`reasoning_len` 3,681）。⇒ **不是能力问题，是预算问题。**
2. **身份混乱是模型权重，不是引擎**：中文 **3/3 Claude/Anthropic**、英文 **2/3 Zhipu/GLM-4.5**、1 次回避，且**跨会话不一致**。接线侧全部单独验过无误：tokenizer 单 token 全绿、chat template 单轮逐字节对、eos 两个都认、`<unk>`=0。⇒ 要重训或换权重才能修。

## 8. 结论与局限

- **结论**：在 **28 题 / 32 请求**的仪器集上，**未见智力退化**。技术面：算数 15/15（含 5 道干扰题）、工具 3/4（安全面 4/4）、SVG 4/4、对话 12/12 干净。
- **局限（必须一起读）**：
  - **28 题不够做统计判据**：15 题算数只有 **±13 点**分辨率；要判"小退化"需 **≥400 题**。
  - **身份认知是硬伤**：技术能力好、身份坏 —— 属微调数据污染，**重训或换权重才能修**。
  - **`scored` 未填**：对错由本轮人工依 `expect` 判定（每条判定依据已列在 §4/§5/§6）。
  - **未验**：多轮对话的长期一致性、长上下文下的能力保持、工具链端到端（本报告只测"是否发出合法 tool_calls"，没有接真实后端执行）。

## 9. 复现资料

| 文件（实验目录下） | 内容 |
|---|---|
| `_raw/intel-test-v1.jsonl` | 题集（28 题，含 `label/kind/lang/difficulty/prompt/expect/max_tokens`） |
| `_raw/intel-b1-chat.json` | 对话 12 条原始（含 `reasoning_len` / `usage` / `wall_s`） |
| `_raw/intel-b1-arith.json` | 算数 15 条原始（含模型原文与 `answer`） |
| `_raw/intel-b1-tool.json` | 工具 4 条原始（含 `tool_calls` 的 name/arguments） |
| `_raw/intel-b1-svg.json` / `_raw/intel-b1-svg-8192.json` | SVG 两档原始 |
| `_raw/intel-b1-svg-only.svg` / `_raw/intel-b1-svg-8192.html` | SVG 产物 |
| `_raw/intel-console.log` | 本轮 serve 侧控制台（请求级耗时） |
