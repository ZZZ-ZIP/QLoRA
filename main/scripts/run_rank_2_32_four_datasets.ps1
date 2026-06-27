param(
    [int[]]$Ranks = @(2..32),
    [switch]$PrepareData,
    [switch]$SkipExisting,
    [switch]$IncludeBase,
    [switch]$DryRun,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$BaseConfig = "configs\emotion_qlora_four_datasets.yaml"
$ConfigDir = "configs\rank_sweep_four_datasets"
$RunRoot = "runs/rank_sweep_four_datasets"
$ProjectPrefix = "qwen3vl_2b_emotion_qlora_four_datasets"
$ResultDir = "results_paper_extension\rank_2_32_four_datasets"

function Invoke-Step {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Name,
        [Parameter(Mandatory=$true)]
        [scriptblock]$Command
    )

    Write-Host "==> $Name"
    if ($DryRun) {
        Write-Host "DRY RUN: $Command"
        return
    }
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code $LASTEXITCODE)"
    }
}

$rankArgs = @()
foreach ($rank in $Ranks) {
    $rankArgs += "$rank"
}

$limitArgs = @()
if ($Limit -gt 0) {
    $limitArgs = @("--limit", "$Limit")
}

New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

if ($PrepareData -or -not (Test-Path -LiteralPath "data_four_datasets\train.jsonl") -or -not (Test-Path -LiteralPath "data_four_datasets\test.jsonl")) {
    Invoke-Step "prepare CMU-MOSEI/CMU-MOSI/CH-SIMSv2/SIMS" {
        python scripts/prepare_mosei_mosi_chsimsv2.py --config $BaseConfig --datasets MOSEI MOSI ch-simsv2s SIMS --overwrite
    }
}

if ($IncludeBase -and -not (Test-Path -LiteralPath "$ResultDir\base_test_metrics.json")) {
    Invoke-Step "predict base on four-dataset test" {
        python scripts/predict_emotion.py --config $BaseConfig --split test --model_kind base --output_file "$ResultDir\base_test_predictions.jsonl" @limitArgs
    }
    Invoke-Step "evaluate base on four-dataset test" {
        python scripts/evaluate_emotion.py --config $BaseConfig --predictions "$ResultDir\base_test_predictions.jsonl" --output_file "$ResultDir\base_test_metrics.json" --output_dir "$ResultDir\base_standard_eval"
    }
}

Invoke-Step "create rank configs r$($Ranks[0])..r$($Ranks[-1])" {
    python scripts/create_rank_configs.py --base_config $BaseConfig --ranks @rankArgs --output_dir $ConfigDir --run_root $RunRoot --project_prefix $ProjectPrefix
}

foreach ($rank in $Ranks) {
    $config = "$ConfigDir\emotion_qlora_r$rank.yaml"
    $metrics = "$ResultDir\adapter_r${rank}_test_metrics.json"
    $predictions = "$ResultDir\adapter_r${rank}_test_predictions.jsonl"
    $standardEvalDir = "$ResultDir\r$rank"

    if ($SkipExisting -and (Test-Path -LiteralPath $metrics)) {
        Write-Host "==> skip r$rank because $metrics exists"
        continue
    }

    Invoke-Step "train r$rank" {
        python scripts/train_qlora_emotion.py --config $config
    }
    Invoke-Step "predict r$rank" {
        python scripts/predict_emotion.py --config $config --split test --model_kind adapter --output_file $predictions @limitArgs
    }
    Invoke-Step "evaluate r$rank" {
        python scripts/evaluate_emotion.py --config $config --predictions $predictions --output_file $metrics --output_dir $standardEvalDir
    }
}

$plotArgs = @("--metrics_dir", $ResultDir, "--ranks") + $rankArgs + @("--output_dir", "$ResultDir\rank_sweep")
if ($IncludeBase) {
    $plotArgs += @("--include_base", "--base_metrics", "$ResultDir\base_test_metrics.json")
}

Invoke-Step "plot r2-r32 rank sweep" {
    python scripts/plot_rank_sweep.py @plotArgs
}

Write-Host "Done. Results: $ResultDir"
