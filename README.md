# VITS2 za srpski jezik

Eksperimentalni projekat sinteze govora (Text-to-Speech) za srpski jezik zasnovan na arhitekturi **VITS2**. Projekat obuhvata pripremu podataka, fonemizaciju srpskog teksta, treniranje jednogovornih i višegovornih modela i inferenciju.

Kod u folderu `vits2_pytorch` nastao je na osnovu javno dostupne VITS2 implementacije, ali je u velikoj meri prilagođen i proširen za srpski jezik i višegovorničko treniranje.

## Glavne karakteristike

- Fonemizacija srpskog teksta (`serbian_cleaners` u `text/cleaners.py`) pomoću biblioteke `phonemizer` sa eSpeak backend-om (jezik `sr`), sa IPA izlazom i oznakama akcenta
- Transfer learning sa pretreniranog engleskog VITS2 checkpoint-a (`G_64000.pth` / `D_64000.pth`)
- Višegovorničko treniranje sa nulto inicijalizovanim novim slojevima za uslovljavanje govornikom
- Treniranje na 143 govornika iz ParlaSpeech korpusa, a zatim dotreniravanje na manjem skupu od 3 govornika
- Skripte za proveru speaker embedding-a i poređenje izlaza po govornicima

## Struktura projekta

```
VITS2/
├── dokumentacija/              # Dokumentacija projekta
├── fine_tuning/                # Config i resursi za fine-tuning
├── skripte za preprocess/      # Priprema filelist-a za VITS2
└── vits2_pytorch/
    ├── configs/                # JSON konfiguracije eksperimenata
    ├── text/                   # Obrada teksta i srpska fonemizacija
    ├── train_ms*.py            # Varijante skripti za višegovorničko treniranje
    ├── inference_*.py          # Skripte za inferenciju
    ├── preprocess_no_torch*.py # Predobrada teksta bez PyTorch zavisnosti
    └── start_finetune.ps1      # Pokretanje fine-tuning-a (PowerShell)
```

## Eksperimenti

| Konfiguracija | Opis |
|---|---|
| `vits2_srpski_3spk_scratch.json` | 3 govornika, treniranje od nule; govor je dugo ostajao nerazumljiv |
| `vits2_srpski_3spk_seeded.json` | 3 govornika, inicijalizacija sa engleskog `G_64000` / `D_64000`, 100.000 koraka; prvi uspešan višegovornički model |
| `vits2_srpski_143_seeded.json` | 143 ParlaSpeech govornika, isti recept, do 250.000 koraka |
| `vits2_srpski_3spk_from143.json` | 3 govornika, inicijalizacija sa 143-govorničkog modela, tako da nasleđuje naučenu srpsku akustiku |
| `vits2_srpski_ft3.json` | *(dopuniti opis)* |

**Ključni zaključak:** treniranje od nule nije dalo razumljiv govor ni posle desetina hiljada koraka, dok je inicijalizacija sa pretreniranog checkpoint-a uz nulto inicijalizovane slojeve za govornike dovela do stvarnog napretka.

## Okruženje

- Windows + PowerShell
- Python 3.10
- NVIDIA GPU (treniranje rađeno na RTX 3090, 24 GB VRAM)
- eSpeak NG (potreban za `phonemizer`)

### Instalacija

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Napomena: ako sistemska zaštita blokira `pip.exe`, pip uvek može da se pokrene kao `python -m pip`.

## Korišćenje

### Priprema podataka

Filelist-e za treniranje priprema skripta iz foldera `skripte za preprocess/`:

```powershell
python "skripte za preprocess\prepare_vits2_filelists.py"
```

### Treniranje

```powershell
cd vits2_pytorch
python train_ms.py -c configs\vits2_srpski_3spk_from143.json -m srpski_3spk_from143
```

### Inferencija

```powershell
cd vits2_pytorch
python inference_ft_simple.py
```

Putanje do modela, config-a i ulaznog teksta podešavaju se u samoj skripti ili kroz argumente.

## Šta nije uključeno u repozitorijum

Zbog veličine i licenci, sledeće se **ne nalazi** u repozitorijumu:

- Checkpoint-i modela (`*.pth`, 400–550 MB po fajlu)
- Audio dataset (`*.wav`)
- TensorBoard logovi
- Python virtuelno okruženje (`venv/`)

## Zahvalnice

- Originalna VITS2 implementacija: *(dopuniti link ka originalnom repozitorijumu)*
- [ParlaSpeech](https://www.clarin.si/) korpus za višegovorničko treniranje
- [phonemizer](https://github.com/bootphon/phonemizer) i [eSpeak NG](https://github.com/espeak-ng/espeak-ng) za fonemizaciju

## Autor

Ivana Kostić
