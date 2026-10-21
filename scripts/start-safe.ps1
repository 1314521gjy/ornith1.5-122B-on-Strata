# Ornith-1.5-122B on Strata · 启动脚本（安全档）
# 用法：
#   .\start-safe.ps1 -Model <gguf 第一片路径> -Exe <strata-poly.exe> -Profile <专家表> -Tokenizer <tokenizer 目录>
# 停止：Ctrl+C，或 taskkill /F /PID <pid>

param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$Exe,
  [Parameter(Mandatory=$true)][string]$Profile,
  [Parameter(Mandatory=$true)][string]$Tokenizer,
  [string]$CudaPath = "",
  [int]$Port = 8123,
  [string]$HostAddr = "127.0.0.1"
)

$ErrorActionPreference = 'Stop'

# --- CUDA DLL（在 bin\x64，不在 bin；缺了会 exit 0xC0000135 且日志一字不写）---
if ($CudaPath -ne "") {
  $env:PATH = "$CudaPath;$env:PATH"
} elseif (-not (Test-Path (Join-Path $env:PATH.Split(';')[0] 'cudart64_13.dll'))) {
  Write-Warning "未指定 -CudaPath。若启动报 exit 0xC0000135，请把 CUDA 的 bin\x64 加进 PATH。"
}

# --- 前置断言 ---
foreach ($p in @($Model, $Exe, $Profile)) {
  if (-not (Test-Path $p)) { throw "找不到: $p" }
}
if (-not (Test-Path (Join-Path $Tokenizer 'merges.txt'))) { throw "tokenizer 目录不完整: $Tokenizer" }

$os = Get-CimInstance Win32_OperatingSystem
$freeGiB = [math]::Round($os.FreePhysicalMemory / 1024 / 1024, 1)
if ($freeGiB -lt 60) { Write-Warning "空闲 RAM 只有 $freeGiB GiB，pin 43 GiB 可能失败。" }

try {
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | ForEach-Object { Write-Host "GPU: $_" }
} catch { Write-Warning "nvidia-smi 不可用" }

# --- 环境（这几条是收益来源，别删）---
$env:STRATA_PIN_ARENA_GIB      = "44"      # 48 层全 pin → prefill +55.4%
$env:STRATA_STAGE_RING2        = "0"       # T8 第二环（负收益，必须关）
$env:STRATA_MAX_PROMPT_BATCH   = "8192"    # 必须 8192；4096 = 空转
$env:STRATA_PROFILE_DECODE_ONLY= "1"
$env:STRATA_AHEAD_EXACT        = "1"
$env:STRATA_SCRATCH_TILE       = "1"
$env:STRATA_PROFILE_AUTO_LEARN = "0"
# 注意：绝不设 STRATA_DET_WINDOW（强制满仓投机，+6% → −24%）

# --- 启动 HTTP 层（serve\server.py 拉起引擎）---
$serverPy = Join-Path (Split-Path $Exe) "..\serve\server.py"
if (-not (Test-Path $serverPy)) { $serverPy = Join-Path (Split-Path $Exe) "..\..\serve\server.py" }
if (-not (Test-Path $serverPy)) { throw "找不到 serve\server.py，请把本仓库的 serve\ 放在引擎树同级。" }

$args = @(
  $serverPy,
  "--exe", $Exe,
  "--model", $Model,
  "--tokenizer", $Tokenizer,
  "--max-context", "32768",
  "--prompt-batch", "4800",
  "--prompt-cache-every", "0",
  "--window", "4",
  "--lookup", "3",
  "--pcie-frac", "0",
  "--stage-mib", "2048",
  "--threads", "16",
  "--expert-profile", $Profile,
  "--host", $HostAddr,
  "--port", "$Port"
)

Write-Host "启动中：$Port ..." -ForegroundColor Cyan
& python @args
