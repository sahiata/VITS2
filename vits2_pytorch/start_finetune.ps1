# start_finetune.ps1
# Priprema i pokretanje FINE-TUNINGA iz G_400000 (ParlaSpeech, 143 govornika)
# na tvoja 3 govornika. Ako bilo koja provera ne prodje, staje PRE treninga.
#
# Pokretanje:
#   powershell -ExecutionPolicy Bypass -File .\start_finetune.ps1

$ErrorActionPreference = "Stop"

$root = "C:\VITS2\vits2_pytorch"
$src  = "$root\logs\srpski_multispeaker_143"
$dst  = "$root\logs\srpski_ft3"
$step = 400000

Write-Host "=== 1) Provera polaznih checkpointa ===" -ForegroundColor Cyan
foreach ($p in @("G", "D")) {
    $f = "$src\$($p)_$step.pth"
    if (-not (Test-Path $f)) { throw "NEDOSTAJE polazni fajl: $f" }
    Write-Host "  OK  $f"
}
# DUR_*.pth je opcion: train_ms.py trenutno ne pravi duration diskriminator (net_dur_disc=None)
$durFile = "$src\DUR_$step.pth"
if (Test-Path $durFile) {
    Write-Host "  OK  $durFile (opciono)"
} else {
    Write-Host "  (DUR_$step.pth ne postoji - nije potreban, duration diskriminator se ne koristi)"
}

Write-Host "=== 2) Provera da su nove verzije train_ms.py i configa na mestu ===" -ForegroundColor Cyan
if (-not (Select-String -Path "$root\train_ms.py" -Pattern "require_checkpoint" -Quiet)) {
    throw "train_ms.py NIJE nova verzija (fali 'require_checkpoint'). Zameni ga poslednjom verzijom."
}
if (-not (Test-Path "$root\configs\vits2_srpski_ft3.json")) {
    throw "Nema $root\configs\vits2_srpski_ft3.json"
}
if (-not (Select-String -Path "$root\configs\vits2_srpski_ft3.json" -Pattern "require_checkpoint" -Quiet)) {
    throw "vits2_srpski_ft3.json NIJE nova verzija (fali 'require_checkpoint')."
}
Write-Host "  OK  train_ms.py i vits2_srpski_ft3.json su nove verzije"

Write-Host "=== 3) Priprema foldera $dst ===" -ForegroundColor Cyan
if (Test-Path $dst) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    try {
        Rename-Item $dst "srpski_ft3_backup_$stamp"
    } catch {
        throw "Ne mogu da preimenujem stari srpski_ft3 (verovatno trening jos radi). Prekini ga sa Ctrl+C pa pokreni ponovo."
    }
    Write-Host "  Stari folder sacuvan kao srpski_ft3_backup_$stamp"
}
New-Item -ItemType Directory -Path $dst | Out-Null

foreach ($p in @("G", "D")) {
    Copy-Item "$src\$($p)_$step.pth" $dst
}
if (Test-Path $durFile) { Copy-Item $durFile $dst }
Write-Host "  Kopirano u ${dst}:"
Get-ChildItem $dst | Format-Table Name, Length -AutoSize

Write-Host "=== 4) Pokretam fine-tuning ===" -ForegroundColor Cyan
Write-Host "  U prvim linijama ispisa MORA da pise:  >>> NASTAVLJAM od checkpointa: epoha ..." -ForegroundColor Yellow
Set-Location $root
& "C:\VITS2\venv\Scripts\Activate.ps1"
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
python train_ms.py -c configs/vits2_srpski_ft3.json -m srpski_ft3
