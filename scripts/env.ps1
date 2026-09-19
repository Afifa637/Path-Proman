# Tier A / Tier B shared environment.  Dot-source before any run:  . .\scripts\env.ps1
#
# Fixes the cp1252 Bangla crash (PLAN.md §4.1) and keeps any Hugging Face cache
# out of the system drive.  HF_HOME defaults into the repo's gitignored
# hf_cache/ so a fresh clone works anywhere; set it yourself beforehand to put
# the cache on a data drive:
#
#   $env:HF_HOME = "E:\NLP\hf_cache"; . .\scripts\env.ps1

$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $repo "src"

if (-not $env:HF_HOME) {
    $env:HF_HOME = Join-Path $repo "hf_cache"
}
if (-not (Test-Path $env:HF_HOME)) {
    New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
}

Write-Host "bnqa env ready - PYTHONPATH=$env:PYTHONPATH  HF_HOME=$env:HF_HOME"
