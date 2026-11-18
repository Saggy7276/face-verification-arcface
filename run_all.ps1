# Runs the full pipeline end to end (Windows PowerShell). Stops at the first failing step.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$steps = @(
    "dataset_preparation.py --overwrite",
    "train.py",
    "generate_pairs.py",
    "evaluate_pairs.py",
    "roc_analysis.py",
    "gallery_probe.py",
    "make_report.py"
)
foreach ($s in $steps) {
    Write-Host "`n=== python $s ===" -ForegroundColor Cyan
    python @($s -split ' ')
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $s" }
}
