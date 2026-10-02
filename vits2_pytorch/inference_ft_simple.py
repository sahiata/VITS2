## VCTK
import glob
import os
import re
import sys
import time
import types

# zaobilazi Application Control blokadu na 'regex' paketu (isti trik kao u treningu)
if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None
    sys.modules["segments"] = _fake_segments

import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


# ============================ OVDE MENJAS ============================

TEXT = "Dobar dan, ovo je probni test srpskog glasa. Da li me dobro čuješ?"

SPK_IDS = [0, 1, 2]       # koji govornici (tvoji speaker0, speaker1, speaker2)

# "latest" = najnoviji G_*.pth, jedan broj (npr. 40000), ili LISTA za poredjenje
# odjednom (npr. [30000, 40000, 50000, 60000, 70000]) - tako se za svaki
# checkpoint napravi poseban fajl i moze da se poredi napredak.
CHECKPOINT = [30000, 40000, 50000, 60000, 70000]

CONFIG_PATH = "./configs/vits2_srpski_ft3.json"
LOGS_DIR = "./logs/srpski_ft3"
OUTPUT_DIR = "./ft_out"

NOISE_SCALE = 0.667       # manje (npr. 0.3) = stabilnije, vise = raznovrsnije
LENGTH_SCALE = 1.0        # veca vrednost = sporiji govor
USE_CPU = False           # True ako GPU zauzima trening i javi 'out of memory'

# =====================================================================


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def ckpt_step(path):
    m = re.search(r"G_(\d+)\.pth$", path)
    return int(m.group(1)) if m else -1


# --- izbor checkpointa (jedan ili vise) ---
all_ckpts = sorted(glob.glob(os.path.join(LOGS_DIR, "G_*.pth")), key=ckpt_step)
if not all_ckpts:
    print(f"Nema G_*.pth u {LOGS_DIR}")
    sys.exit(1)

if CHECKPOINT == "latest":
    # najnoviji po VREMENU IZMENE (u folderu moze da stoji i polazni G_400000 sa vecim brojem)
    MODEL_PATHS = [max(all_ckpts, key=os.path.getmtime)]
elif isinstance(CHECKPOINT, (list, tuple)):
    MODEL_PATHS = []
    for c in CHECKPOINT:
        p = os.path.join(LOGS_DIR, f"G_{int(c)}.pth")
        if not os.path.isfile(p):
            print(f"[preskacem] ne postoji {p}")
            continue
        MODEL_PATHS.append(p)
    if not MODEL_PATHS:
        print(f"Nijedan trazeni checkpoint ne postoji. Dostupni: {[ckpt_step(c) for c in all_ckpts]}")
        sys.exit(1)
else:
    p = os.path.join(LOGS_DIR, f"G_{int(CHECKPOINT)}.pth")
    if not os.path.isfile(p):
        print(f"Ne postoji {p}. Dostupni: {[ckpt_step(c) for c in all_ckpts]}")
        sys.exit(1)
    MODEL_PATHS = [p]

print(f"Tekst: {TEXT}")
print(f"Checkpointi: {[ckpt_step(c) for c in MODEL_PATHS]}")

device = "cpu" if USE_CPU else "cuda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

hps = utils.get_hparams_from_file(CONFIG_PATH)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).to(device)
_ = net_g.eval()

stn_tst = get_text(TEXT, hps)

for MODEL_PATH in MODEL_PATHS:
    step = ckpt_step(MODEL_PATH)
    saved_at = time.strftime("%H:%M", time.localtime(os.path.getmtime(MODEL_PATH)))
    print(f"\n--- checkpoint G_{step}.pth (sacuvan u {saved_at}) ---")
    _ = utils.load_checkpoint(MODEL_PATH, net_g, None)

    for spk_id in SPK_IDS:
        torch.manual_seed(1234)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            sid = torch.LongTensor([spk_id]).to(device)
            audio = (
                net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    sid=sid,
                    noise_scale=NOISE_SCALE,
                    noise_scale_w=0.0,
                    length_scale=LENGTH_SCALE,
                )[0][0, 0]
                .data.cpu()
                .float()
                .numpy()
            )
        out_path = os.path.join(OUTPUT_DIR, f"ft{step}_spk{spk_id}.wav")
        write(data=audio, rate=hps.data.sampling_rate, filename=out_path)
        print(f"Sacuvano: {out_path}")

print("\nGotovo. Poredi fajlove istog spk-a kroz razlicite ft<korak>_ vrednosti.")
