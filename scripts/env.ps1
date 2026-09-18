# Tier A / Tier B shared environment.  Dot-source before any run:  . .\scripts\env.ps1
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null
$env:HF_HOME = "E:\NLP\hf_cache"
$env:PYTHONPATH = "$PSScriptRoot\..\src"
Write-Host "bnqa env ready - PYTHONPATH=$env:PYTHONPATH  HF_HOME=$env:HF_HOME"
