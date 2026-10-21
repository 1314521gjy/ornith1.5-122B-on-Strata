# 预编译引擎（Prebuilt engine）

`strata-poly.exe` — **62.62 MB** — sha256：

```
03885B2402A4CAB3CA8742AE9345F66CB9EE1881A5A20E88C763A5A07B34E26B
```

## 编译环境

| 项 | 值 |
|---|---|
| CUDA | **13.3** |
| MSVC | **14.51**（Visual Studio 2022） |
| CMake / Ninja | 4.x / Release |
| 源码 | 本仓库 `src/poly/`（基线 **Strata `v0.1.44.1-poly.3`**，commit `eb5ef118`） |
| 目标架构 | **sm_89**（RTX 40 系；实测 RTX 4080 SUPER 32 GB） |

## 依赖（必须自己装）

| DLL | 来源 | 大小 |
|---|---|---|
| `cudart64_13.dll` | CUDA 13.3 | 0.53 MB |
| `cublas64_13.dll` | CUDA 13.3 | 49.47 MB |
| `cublasLt64_13.dll` | CUDA 13.3 | **438.98 MB** |

**DLL 在 `<cuda>\bin\x64`，不在 `<cuda>\bin`** —— 这条踩过：缺了会 `exit 0xC0000135` 且**日志一字不写**。

**本仓库不分发 DLL**（438 MB 太大）。装 **CUDA 13.3 Runtime** 即可。

## 硬性要求

- **显存 ≥ 32 GB** —— 24 GB 卡跑不了（专家缓存会被挤到 0）
- **RAM ≥ 64 GiB**（pin 43.03 GiB 锁页 + 模型 mmap）—— 建议 96 GiB
- **Windows x64**（本 exe 是 Windows 构建；Linux 需自行编译）

## 推荐：自己编译

别人的 CUDA 版本、驱动、显卡架构和你可能不同 —— **自己编的 exe 更匹配你的机器**。
见 [`../DEPLOY.md`](../DEPLOY.md) §3，约 76 秒（16 线程）。

## 校验

```powershell
(Get-FileHash .\strata-poly.exe -Algorithm SHA256).Hash
# 期望 03885B2402A4CAB3CA8742AE9345F66CB9EE1881A5A20E88C763A5A07B34E26B
```

## 怎么用

```powershell
# 前置：CUDA DLL 进 PATH
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;$env:PATH"

# 用仓库自带脚本（它会拉起 serve\server.py 再由它启动本 exe）
.\scripts\start-safe.ps1 -Model <第一片.gguf> -Exe .\bin\strata-poly.exe `
                         -Profile .\profile\expert-profile-ornith122b.txt -Tokenizer .\tokenizer
```
