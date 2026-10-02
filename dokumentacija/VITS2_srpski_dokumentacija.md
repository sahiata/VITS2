# VITS2 Srpski Multispeaker TTS — Dokumentacija procesa

Ovaj dokument sumira sve korake preduzete za pripremu i pokretanje VITS2 treninga
za srpski jezik, multispeaker (3 govornika), sa transfer learning-om sa engleskog
LJSpeech checkpoint-a. Okruženje: Windows, PowerShell, VS Code, RTX 3090 (24GB VRAM).

---

## 0. Zašto VITS2 umesto Piper-a

Piper (VITS1 arhitektura) je pokazao tri konkretna problema u prethodnim treninzima:
- Neujednačen tempo govora (prva polovina rečenice izgovorena brže od druge)
- Slaba akcentuacija (posledica same VITS1 arhitekture + ograničenog dataset-a)
- Potreba za većim vokabularom za nove/neviđene reči

VITS2 direktno cilja na prva dva problema kroz poboljšan generator i dodatnu
adversarijalnu komponentu za trajanje fonema (duration discriminator), dok VITS2
ostaje arhitekturno najbliži Piper-u (isti eSpeak-ng front-end, sličan hardverski
budžet) — manji rizik od potpune promene arhitekture (npr. StyleTTS2), koji uz to
nema podršku za srpski/slovenske jezike.

Repo izabran: **p0p4k/vits2_pytorch** — najčistija dostupna neslužbena
VITS2 implementacija (Kakao Enterprise, originalni autori, nikad nisu objavili
zvaničan kod).

---

## 1. Instalacija repozitorijuma

**Link:** https://github.com/p0p4k/vits2_pytorch

**Preduslov — Git nije bio instaliran, instaliran preko winget-a (PowerShell, po
mogućstvu kao administrator):**
```powershell
winget install --id Git.Git -e --source winget
```
*Napomena: posle instalacije, PowerShell/VS Code terminal mora biti potpuno
zatvoren i ponovo otvoren da bi PATH promena stupila na snagu — ovo se pokazalo
kao ponavljajući problem tokom cele sesije (isto važi za sve `$env:` promenljive
i novoinstalirane alate).*

**Kloniranje:**
```powershell
cd C:\VITS2
git clone https://github.com/p0p4k/vits2_pytorch
cd vits2_pytorch
```

---

## 2. Virtuelno okruženje (venv)

**Zašto:** VITS2 ima specifične verzije zavisnosti (stariji PyTorch, Cython,
numpy) koje mogu da se sudare sa drugim projektima (npr. Piper environment) ako
se instaliraju globalno.

```powershell
cd C:\VITS2
python -m venv venv
venv\Scripts\Activate.ps1
```
*Python verzija: 3.10 (repo zahteva >= 3.10; testiran sa starijim PyTorch-om
1.13.1, pa je 3.10 sigurniji izbor od najnovijeg 3.12 zbog kompatibilnosti
zavisnosti).*

---

## 3. PyTorch sa CUDA podrškom

**Zašto pre requirements.txt:** ako se instalira posle, `requirements.txt` može
povući CPU-only verziju PyTorch-a.

GPU: RTX 3090, drajver podržava CUDA do 13.3. Izabran CUDA 12.6 build kao
najstabilniji/najtestiraniji u kombinaciji sa starijim kodom u repo-u:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Poznat problem:** `flit_core` greška pri instalaciji `typing_extensions` (pip
ograničen samo na PyTorch server preko `--index-url`, ne može da nađe pomoćni
build paket). Rešeno:
```powershell
pip install --upgrade typing_extensions
```
pa ponovljena glavna komanda.

**Provera:**
```powershell
python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
```

---

## 4. Ostale zavisnosti

```powershell
pip install -r requirements.txt
```

---

## 5. Kompajliranje monotonic_align (C/Cython ekstenzija)

**Zašto je ovo bilo komplikovano:** obična PowerShell sesija ne zna gde su MSVC
build alati (`cl.exe`), čak i kad su Visual Studio Build Tools instalirani.

**Rešenje — koristiti poseban terminal:**
1. Otvoriti **"x64 Native Tools Command Prompt for VS"** iz Start menija (NE
   obični "Developer Command Prompt" — taj daje 32-bit `cl.exe`, treba nam x64).
2. U tom terminalu:
```
cd C:\VITS2\vits2_pytorch
..\venv\Scripts\activate.bat
cl
```
   (proveriti da piše "for x64", ne "for x86")
3. Kompajliranje:
```
cd monotonic_align
python setup.py build_ext --inplace
```

**Napomena:** ovaj poseban environment treba SAMO za ovaj jedan korak. Posle
uspešnog kompajliranja (`.pyd` fajl je fizički kreiran), može se slobodno nastaviti
rad iz običnog VS Code terminala.

---

## 6. Priprema srpskog dataset-a — konverzija u VITS2 format

**Zašto:** Piper metadata.csv format je `relativna_putanja|ime_govornika|tekst`
(npr. `speaker2/a97|speaker2|tekst`). VITS2 multispeaker (`train_ms.py`) očekuje
`puna_putanja.wav|broj_govornika|tekst` — govornik mora biti **broj**, ne string,
jer se koristi kao indeks u speaker embedding tabeli.

Napravljen Python skript (`prepare_vits2_filelists.py`) koji:
- Čita postojeći `metadata.csv` (format: `speaker0/a1|speaker0|tekst`)
- Izvlači broj govornika iz imena foldera (`speaker0` → `0`, `speaker1` → `1`, itd.)
- Dodaje `.wav` ekstenziju i punu putanju
- Radi **stratifikovanu** train/val/test podelu (90/5/5) po govorniku — ne
  nasumičnu globalno, da se izbegne da neki govornik skoro nestane iz val/test seta
- Upisuje tri odvojena fajla: `vits2_train_filelist.txt`, `vits2_val_filelist.txt`,
  `vits2_test_filelist.txt`

**Rezultat:** 2955 od 2956 linija uspešno konvertovano (1 nedostajući `.wav`
fajl preskočen), raspodeljeno na 3 govornika (901, 1094, 960 rečenica), split
2658/147/150 (train/val/test).

**Bitno:** fajlovi za train/val/test ostaju **odvojeni po nameni**, ali **sva
tri govornika su izmešana unutar svakog fajla** — to je format koji `train_ms.py`
očekuje za multispeaker trening.

Fajlovi prekopirani u `vits2_pytorch\filelists\` (konvencija koju repo koristi za
relativne putanje u config-u).

---

## 7. eSpeak-ng podešavanje za srpski (serbian_cleaners)

**Problem:** `text/cleaners.py` ima samo engleske cleaner funkcije
(`english_cleaners2` itd.), koje pozivaju eSpeak-ng sa `language="en-us"` i, što
je bitnije, uključuju `unidecode()` transliteraciju — što bi **uništilo** srpske
dijakritike (č, ć, š, ž, đ → c, c, s, z, dj), tačno onu informaciju potrebnu za
ispravnu akcentuaciju.

**Rešenje — dodata `serbian_cleaners` funkcija na kraj `text/cleaners.py`:**
```python
def serbian_cleaners(text):
    """Pipeline for Serbian text — preserves diacritics (č, ć, š, ž, đ), punctuation and stress."""
    text = lowercase(text)
    phonemes = phonemize(
        text,
        language="sr",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes
```
Namerno izostavljen `convert_to_ascii()` korak iz razloga gore navedenog.

**Registracija:** nije bila potrebna — `text/__init__.py` koristi
`getattr(cleaners, name)`, dinamički pronalazi funkciju po imenu string-a.

**eSpeak-ng environment problem:** `phonemizer` Python paket ne pronalazi
eSpeak-ng automatski na Windows-u, čak i kad je instaliran. Rešenje — postaviti
environment promenljivu **u SVAKOM novom terminalu** pre pokretanja bilo čega
što koristi `text` modul:
```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
```

**Provera vokabulara:** napravljen test skript
(`test_serbian_phonemes.py`) koji fonemizuje test rečenice sa svim "rizičnim"
srpskim glasovima (č, ć, š, ž, đ, dž, nj, lj) i proverava da li svaki proizvedeni
IPA simbol postoji u `text/symbols.py` vokabularu — jer `cleaned_text_to_sequence()`
**tiho** (bez greške) preskače simbole koji nisu u vokabularu, što bi moglo
neprimetno da "osakati" trening podatke. **Rezultat: svi testirani fonemi
pokriveni**, vokabular (177 simbola, širok multi-jezički IPA superset) je
dovoljan za srpski bez izmena.

---

## 8. Fonemizacija dataset-a (preprocess.py)

**Komanda (mora imati SVE argumente eksplicitno — bez njih, skript pada na
podrazumevane LJSpeech/engleske vrednosti):**
```powershell
python preprocess.py --text_index 2 --text_cleaners serbian_cleaners --filelists filelists/vits2_train_filelist.txt filelists/vits2_val_filelist.txt filelists/vits2_test_filelist.txt
```
`--text_index 2` jer format ima 3 kolone (`putanja|sid|tekst`), tekst je na
indeksu 2 (brojanje od 0), ne 1 kao kod LJSpeech dvokolonskog formata.

Upozorenja `words count mismatch` tokom izvršavanja su bezopasna — posledica
`preserve_punctuation=True` opcije, ne greška u podacima (potvrđeno vizuelnom
proverom generisanih fonema).

**Rezultat:** tri `.cleaned` fajla, spremna za trening config.

---

## 9. Checkpoint za transfer learning

**Zašto transfer learning:** isti princip kao kod Piper-a (lessac/bulgaria
checkpoint) — ubrzava konvergenciju, model ne uči akustiku od nule.

**Izabran checkpoint:** zvanični **LJSpeech-no-sdp** (64k koraka, engleski,
single-speaker), sa Google Drive-a:
https://drive.google.com/drive/folders/1U-1EqBMXqmEqK0aUhbCJOquowbvKkLmc?usp=sharing

**Zašto LJSpeech umesto ruskog Natasha checkpoint-a** (alternativa koja je
razmatrana, isti repo, isti sample rate 22050Hz): LJSpeech config eksplicitno
potvrđuje `"cleaned_text": true` + `"text_cleaners": ["english_cleaners2"]`, što
znači da model uči na **univerzalnim IPA fonemima**, ne na engleskim slovima —
fonemi su jezički nezavisni, pa transfer na srpski ima smisla. Nije bilo moguće
sa sigurnošću potvrditi da li ruski checkpoint koristi isti standardni IPA
pristup (naznake u community diskusiji sugerišu prilagođen ruski skup simbola).

**Poređenje config-a (LJSpeech-no-sdp vs. ciljni VCTK/srpski multispeaker
config)** — skoro potpuno poklapanje arhitekture:

| Parametar | LJSpeech-no-sdp | Srpski multispeaker |
|---|---|---|
| `n_speakers` | 0 | 3 |
| `use_spk_conditioned_encoder` | false | true |
| `gin_channels` | (ne postoji) | 256 |
| sve ostalo | identično | identično |

Znači: dekoder, flow slojevi, posterior encoder, duration discriminator, glavnina
tekstualnog enkodera — čisto se prenose. Jedino speaker embedding sloj ostaje
nasumično inicijalizovan (logično, LJSpeech nikad nije "video" više govornika).

**Priprema:**
```powershell
mkdir C:\VITS2\vits2_pytorch\logs\srpski_multispeaker
Copy-Item "C:\Users\User\Downloads\G_64000.pth" "C:\VITS2\vits2_pytorch\logs\srpski_multispeaker\G_0.pth"
```
Diskriminator (`D_*.pth`) nije objavljen za ovaj checkpoint — nije problem,
diskriminator se standardno uči od nule (uobičajena praksa).

---

## 10. Config fajl (configs/vits2_srpski.json)

Napravljen na osnovu `configs/vits2_vctk_base.json` (pravi VITS2 multispeaker
template), sa izmenama:
- `training_files` / `validation_files` → putanje na `.cleaned` fajlove
- `text_cleaners: ["serbian_cleaners"]`
- `n_speakers: 3`
- `batch_size: 16` (smanjeno sa originalnih 64 — VCTK config pravljen za veći
  dataset/klaster; manji batch smanjuje rizik od "out of memory" na RTX 3090
  sa manjim dataset-om)

---

## 11. Ispravke u train_ms.py (Windows-specifični problemi)

Ovo je bio najduži deo procesa — nekoliko različitih grešaka, svaka je otkrivena
tek kad je prethodna ispravljena.

### 11.1 NCCL greška
```
RuntimeError: Distributed package doesn't have NCCL built in
```
**Zašto:** NCCL (komunikacioni backend za multi-GPU) je Linux-only, ne postoji
za Windows PyTorch build. Nepotreban za jednu GPU.

**Ispravka** u `dist.init_process_group(...)`:
```python
backend="nccl"  →  backend="gloo"
```

### 11.2 Tih pad (access violation, exit code 0xC0000005)
Pad bez Python traceback-a, dešavao se dosledno posle DDP inicijalizacije
(prvi backward prolaz).

**Pokušaj 1 (nije bio pravi uzrok, ali nije ni štetio):** smanjen
`num_workers=8` → `num_workers=0` u oba `DataLoader` poziva — pretpostavka da je
ugnježdeni multiprocessing (mp.spawn + DataLoader workers) problem. Pad se i
dalje dešavao.

**Pravi uzrok:** `gloo` backend + CUDA tenzori tokom DDP komunikacije nepouzdano
rade na Windows-u. Pošto je trening na jednoj GPU, DDP sloj je nepotreban u
potpunosti.

**Ispravka 1 — zaobiđen `mp.spawn`** u `main()`:
```python
hps = utils.get_hparams()
if n_gpus <= 1:
    run(0, n_gpus, hps)
else:
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))
```

**Ispravka 2 — zaobiđen `DDP()` wrapper** (glavni uzrok):
```python
if n_gpus > 1:
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if net_dur_disc is not None:
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)
else:
    # Bez DDP-a za single-GPU trening (izbegava gloo/CUDA problem na Windows-u).
    # Ostatak koda poziva net_g.module / net_d.module, pa dodajemo .module
    # kao referencu na sam objekat (object.__setattr__ zaobilazi nn.Module
    # mehanizam koji bi inače registrovao ovo kao pod-modul i napravio
    # beskonačnu petlju reference na samog sebe).
    object.__setattr__(net_g, "module", net_g)
    object.__setattr__(net_d, "module", net_d)
    if net_dur_disc is not None:
        object.__setattr__(net_dur_disc, "module", net_dur_disc)
```
**Napomena — greška pri prvoj primeni:** `else` je slučajno bio uvučen za
jedan nivo previše (vezan za unutrašnji `if net_dur_disc is not None:` umesto
za spoljašnji `if n_gpus > 1:`), zbog čega se kod za single-GPU nikad nije
izvršavao. Ispravljeno poravnanjem `else:` na isti nivo kao `if n_gpus > 1:`.

Posle ove ispravke, trening je konačno krenuo stabilno.

### 11.3 num_workers vraćen na razumnu vrednost
Pošto pravi uzrok pada nije bio DataLoader, `num_workers=0` je ostavljao
učitavanje podataka sekvencijalno (CPU čeka na svaki fajl pojedinačno, GPU
stoji prazan između batch-eva — vidljivo kao "talasast" GPU-Util grafikon u
Task Manageru, 55-67% sa oštrim padovima umesto ravne linije blizu 100%).
Vraćeno na `num_workers=4`.

---

## 12. Pokretanje treninga

```powershell
cd C:\VITS2\vits2_pytorch
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
python train_ms.py -c configs/vits2_srpski.json -m srpski_multispeaker
```

**Prva epoha je spora** (~20-25s po koraku umesto par sekundi) — `TextAudioSpeakerLoader`
računa mel-spektrogram za svaki `.wav` fajl pri prvom čitanju i kešira ga na
disk kao `<ime>.mel.pt` pored originala. Naredne epohe čitaju gotov keš i
značajno su brže.

Checkpoint-i (`G_*.pth`, `D_*.pth`, `DUR_*.pth`) čuvaju se u
`logs\srpski_multispeaker\` na svakih `eval_interval` (1000) koraka — kod
automatski zadržava samo poslednja 3 seta (`remove_old_checkpoints`).
Za samo generisanje govora, bitan je isključivo `G_*.pth` fajl; `D_*.pth` i
`DUR_*.pth` služe samo za nastavak treninga.

Pri svakom restartu treninga (posle grešaka tokom debagovanja), kod automatski
učitava **najnoviji** dostupan `G_*.pth`/`D_*.pth` checkpoint i nastavlja
odatle — napredak se ne gubi između restartova.

---

## 13. Generisanje audio uzorka (inferenca)

Fajl `inference_ms.py` — menjaju se promenljive direktno na vrhu fajla (nema
argumenata komandne linije):
```python
CONFIG_PATH = "./configs/vits2_srpski.json"
MODEL_PATH = "./logs/srpski_multispeaker/G_45000.pth"  # najnoviji checkpoint
TEXT = "Dobar dan, ovo je prvi test srpskog glasa."
SPK_ID = 0  # 0, 1, ili 2 — koji govornik
OUTPUT_WAV_PATH = "test_speaker0.wav"
```
`get_text()` funkcija u ovom skriptu poziva `text_to_sequence()`, koja
automatski primenjuje `hps.data.text_cleaners` iz config-a (`serbian_cleaners`)
— sirov srpski tekst se unosi direktno, fonemizacija je automatska.

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
python inference_ms.py
```

---

## 14. Praćenje napretka (TensorBoard)

```powershell
tensorboard --logdir logs/srpski_multispeaker
```
Otvoriti `http://localhost:6006` u browseru.

**Šta gledati:**
- **Scalars tab → `loss/g/mel`, `loss/g/dur`** — glavni pokazatelji stvarnog
  napretka, trebalo bi da opadaju tokom treninga.
- **Images tab → `all/attn`** — mapa poravnanja teksta i zvuka. Bitan je
  **oblik** linije (glatka dijagonala = alignment naučen), ne brojevi na osama
  (to je samo dužina te konkretne nasumično izabrane test rečenice).
- **Images tab → `all/mel` i `gen/mel`** — mel-spektrogrami; oštrina/jasnoća
  harmonijskih pruga pokazuje kvalitet zvuka. Mutna slika = "bebasto" mumljanje
  koje se čuje; postaje oštrija sa daljim treningom.

**Status na ~45-47k koraka (~293 epohe):** `all/attn` već pokazuje jasnu, glatku
dijagonalu (alignment je naučen) — ali `all/mel`/`gen/mel` su još mutni, zvuk
zvuči nerazgovetno ("kao beba"). Ovo je očekivano — alignment se obično formira
pre oštrine/jasnoće detalja zvuka. VITS treninzi tipično zahtevaju
80.000-150.000+ koraka za jasno razumljiv govor.

**Za ubrzanje treninga (razmatrano, nije još primenjeno):**
`"fp16_run": true` u config-u — mixed precision, na RTX 3090 (ima Tensor Cores)
tipično 1.5-2x ubrzanje uz nizak rizik (i smanjuje memorijski otisak, koji je
trenutno na granici, 22.1/24GB).

---

## Sledeći koraci (nije još urađeno)

- Nastaviti trening dok `loss/g/mel` ne pokaže jasan pad i `gen/mel` postane
  oštriji
- Razmotriti `fp16_run: true` za ubrzanje
- Kasnije: probati **single-speaker** trening (`train.py` umesto `train_ms.py`),
  moguće sa ruskim Natasha checkpoint-om kao osnovom, ili sa jednim govornikom
  izdvojenim iz postojećeg dataset-a
