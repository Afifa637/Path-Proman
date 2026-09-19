# VC-11 — regenerate everything from a clean checkout.
#
#   . .\scripts\env.ps1
#   .\scripts\run_all.ps1              # the whole Tier A pipeline
#   .\scripts\run_all.ps1 -Quick       # a small pass, for checking the wiring
#   .\scripts\run_all.ps1 -SkipFetch   # corpora already downloaded
#
# corpus -> index -> models -> evaluation -> every table and figure.
# Nothing here is optional and nothing is hand-edited afterwards: if a number
# in the report is not produced by this script, it does not belong in the
# report (VC-12).
#
# Wall clock on a 16-core CPU: about 2 hours end to end, of which the 50k
# retrieval sweep is roughly half.  -Quick finishes in ten minutes and proves
# the wiring without pretending to be an evaluation.

[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$SkipFetch,
    [int]$Size = 50000
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

$readerTrain = if ($Quick) { 200 } else { 3000 }
$readerEval  = if ($Quick) { 100 } else { 800 }
$verifyEval  = if ($Quick) { 100 } else { 800 }
$cfN         = if ($Quick) { 5 }   else { 30 }
$idx         = if ($Quick) { 10000 } else { $Size }

function Step([string]$label, [scriptblock]$body) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    Write-Host "  $label" -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor DarkGray
    $t0 = Get-Date
    & $body
    if ($LASTEXITCODE -ne 0) { throw "$label failed with exit code $LASTEXITCODE" }
    $dt = (Get-Date) - $t0
    Write-Host ("  [{0:mm\:ss}] {1}" -f $dt, $label) -ForegroundColor DarkGreen
}

Step "T0  environment" { python -u -m bnqa.envinfo }

if (-not $SkipFetch) {
    Step "T0  fetch corpora (~550 MB, hash-pinned)" { python -u -m bnqa.data.fetch }
}

Step "T1  BanglaRQA ingest + leakage gate + alignment audit" { python -u -m bnqa.data.banglarqa }
Step "T2  NCTB textbook ingest"                              { python -u -m bnqa.data.nctb_text }
Step "T3a Wikipedia length-matched distractors"              { python -u -m bnqa.data.wiki }
Step "T3  index assembly (nested 10k / 50k)"                 { python -u -m bnqa.data.index_build }
Step "T3  corpus audit"                                      { python -u -m bnqa.data.audit }
Step "T6b lexical resource probe"                            { python -u -m bnqa.eval.lexicon_probe }

if ($Quick) {
    Step "T5/T7 sparse retrieval (quick)" {
        python -u scripts/run_retrieval.py --size 10000 --queries 150
    }
} else {
    Step "T5/T7 sparse retrieval, both index sizes, tuned on val" {
        python -u scripts/run_retrieval.py
    }
}

Step "T7  topic structure (K-Means + silhouette)" {
    python -u -m bnqa.eval.topics
}

Step "T10 reader: question type, span ranker, feature table" {
    python -u scripts/run_reader.py --size $idx --train $readerTrain --eval $readerEval
}

Step "T11 verifier: BanglaVerify, S3, fusion, calibration, conformal" {
    python -u scripts/run_verify.py --size $idx --eval $verifyEval
}

Step "V3  counterfactual corpus test (VC-7)" {
    python -u scripts/run_counterfactual.py --n $cfN
}

Step "T13 receipts, audit sheet, model card" {
    python -u scripts/make_report_assets.py --size $idx
}

Step "V1  re-verify every receipt offline (VC-3)" {
    python -u scripts/verify_receipt.py
}

Step "tests: every invariant" { python -u -m pytest -q }

Write-Host ""
Write-Host "Done.  reports/results.json, reports/tables/ and reports/figures/ are current." -ForegroundColor Green
Write-Host "Config hash:" -NoNewline
python -c "import sys; sys.path.insert(0,'src'); from bnqa.config import config_hash; print(' ' + config_hash())"
