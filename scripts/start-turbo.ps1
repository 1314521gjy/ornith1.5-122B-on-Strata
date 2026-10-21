# Ornith-1.5-122B on Strata · 启动脚本（turbo 有损档）
# ⚠️ 有损：每 token 跳过约 14.2 次专家使用，输出会变（上游测 KL≈0.0058）。默认不要用。
# 用法与 start-safe.ps1 相同，多一个 -Ack 参数用于明确确认"我知道它有损"。

param(
  [Parameter(Mandatory=$true)][string]$Model,
  [Parameter(Mandatory=$true)][string]$Exe,
  [Parameter(Mandatory=$true)][string]$Profile,
  [Parameter(Mandatory=$true)][string]$Tokenizer,
  [switch]$Ack,
  [string]$CudaPath = "",
  [int]$Port = 8124,
  [string]$HostAddr = "127.0.0.1"
)

if (-not $Ack) {
  Write-Host "turbo 档是有损的：每 token 跳过约 14.2 次专家使用、输出会变。" -ForegroundColor Yellow
  Write-Host "确认请加 -Ack 重跑。" -ForegroundColor Yellow
  exit 1
}

$ErrorActionPreference = 'Stop'

if ($CudaPath -ne "") { $env:PATH = "$CudaPath;$env:PATH" }
foreach ($p in @($Model, $Exe, $Profile)) { if (-not (Test-Path $p)) { throw "找不到: $p" } }
if (-not (Test-Path (Join-Path $Tokenizer 'merges.txt'))) { throw "tokenizer 目录不完整: $Tokenizer" }

$env:STRATA_PIN_ARENA_GIB       = "44"
$env:STRATA_STAGE_RING2         = "0"
$env:STRATA_MAX_PROMPT_BATCH    = "8192"
$env:STRATA_PROFILE_DECODE_ONLY = "1"
$env:STRATA_AHEAD_EXACT         = "1"
$env:STRATA_SCRATCH_TILE        = "1"
$env:STRATA_PROFILE_AUTO_LEARN  = "0"

# === turbo 唯一的差别 ===
$env:STRATA_ROUTE_TAIL_SKIP     = "7"     # decode +18%，代价=质量

$serverPy = Join-Path (Split-Path $Exe) "..\serve\server.py"
if (-not (Test-Path $serverPy)) { $serverPy = Join-Path (Split-Path $Exe) "..\..\serve\server.py" }
if (-not (Test-Path $serverPy)) { throw "找不到 serve\server.py" }

$args = @(
  $serverPy,
  "--exe", $Exe, "--model", $Model, "--tokenizer", $Tokenizer,
  "--max-context", "32768", "--prompt-batch", "4800", "--prompt-cache-every", "0",
  "--window", "4", "--lookup", "3", "--pcie-frac", "0",
  "--stage-mib", "2048", "--threads", "16",
  "--expert-profile", $Profile,
  "--host", $HostAddr, "--port", "$Port"
)

Write-Host "启动 turbo 档（有损）：$Port ..." -ForegroundColor Yellow
& python @args
