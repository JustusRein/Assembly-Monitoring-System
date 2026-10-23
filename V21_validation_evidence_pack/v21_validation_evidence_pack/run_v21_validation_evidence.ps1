$ErrorActionPreference = 'Stop'
$Project = 'D:\Thesis\SOFTWARE V2.1 AI Final\Assembly_monitor'
$Python = 'D:\Thesis\SOFTWARE V2.1 AI Final\venv\Scripts\python.exe'
$Dataset = 'D:\Thesis\Blender\synthetic_dataset'
Set-Location $Project
& $Python .\build_v21_validation_evidence.py --dataset-root $Dataset
if ($LASTEXITCODE -ne 0) { throw "Evidence report failed with exit code $LASTEXITCODE" }
