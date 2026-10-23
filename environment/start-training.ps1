param([switch]$VisibleChild)
$ErrorActionPreference = 'Stop'

if (-not $VisibleChild) {
    $hostExecutable = (Get-Process -Id $PID).Path
    $quotedScript = '"' + $PSCommandPath + '"'
    Start-Process -FilePath $hostExecutable -WindowStyle Normal -ArgumentList @(
        '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $quotedScript, '-VisibleChild'
    )
    Write-Host '已打开独立训练 PowerShell 窗口；新窗口会显示条件检查和实时训练进度。'
    return
}

$workspaceRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspaceRoot
. (Join-Path $PSScriptRoot 'activate-ai-cuda.ps1')
$env:DESKVISION_VISIBLE_TRAINING = '1'
Write-Host '即将执行：python -m deskvision.train --run'
python -m deskvision.train --run
if ($LASTEXITCODE -ne 0) {
    Write-Host "训练进程退出代码：$LASTEXITCODE" -ForegroundColor Red
}
