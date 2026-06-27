param(
    [int[]]$Ranks = @(4, 8, 10, 12, 14, 16, 20, 32)
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$rankArgs = @()
foreach ($rank in $Ranks) {
    $rankArgs += "$rank"
}

python scripts/plot_rank_sweep.py `
    --metrics_dir outputs `
    --ranks @rankArgs `
    --auto_discover `
    --include_base `
    --output_dir outputs/all_rank_results

if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate all-rank plot."
}
