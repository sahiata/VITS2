## VCTK
"""
inference_ft.py - generisanje govora iz fine-tuning checkpointa (multi-speaker VITS2)

Podrazumevano: config ./configs/vits2_srpski_ft3.json, checkpointi u ./logs/srpski_ft3,
govornici 0,1,2 (tvoji speaker0/1/2).

Primeri (iz C:\\VITS2\\vits2_pytorch, venv aktivan, PHONEMIZER_ESPEAK_LIBRARY postavljen):

  # NAJKORISNIJE: prvih 5 recenica iz VALIDACIONOG skupa, sa originalnim snimkom pored
  python inference_ft.py --ckpt 30000 --val 5

  # sopstveni tekst za sva tri govornika
  python inference_ft.py --ckpt 30000 --text "Dobar dan, ovo je probni test."

  # najnoviji checkpoint (po vremenu izmene), jedan govornik
  python inference_ft.py --spk 1

  # svi fine-tuning checkpointi (polazni G_400000 se preskace), da ih poredis
  python inference_ft.py --ckpt all --val 3

  # ako GPU zauzima trening i javi 'out of memory'
  python inference_ft.py --ckpt 30000 --val 3 --cpu

Izlaz u ft_out\\ :
  ft<korak>_<oznaka>_spk<id>.wav      - generisano
  <oznaka>_original_spk<id>.wav       - pravi snimak (samo uz --val)
  <oznaka>_spk<id>_tekst.txt          - tekst rečenice (samo uz --val)
Tekst se unosi OBICAN (bez '+' naglasaka) - fonemizaciju radi serbian_cleaners.
"""
import argparse
import glob
import os
import re
import shutil
import sys
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

DEFAULT_TEXT = "Dobar dan, ovo je probni test srpskog glasa. Da li me dobro čuješ?"


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


def ckpt_step(path):
    m = re.search(r"G_(\d+)\.pth$", path)
    return int(m.group(1)) if m else -1


def load_val_items(path, n):
    """Cita prvih n redova SIROVOG validacionog filelist-a: putanja|speaker_id|tekst"""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            items.append((parts[0], int(parts[1]), "|".join(parts[2:])))
            if len(items) >= n:
                break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./configs/vits2_srpski_ft3.json")
    ap.add_argument("--logs", default="./logs/srpski_ft3", help="folder sa G_*.pth")
    ap.add_argument("--ckpt", default="latest", help="'latest' (najnoviji po vremenu), 'all' ili broj koraka")
    ap.add_argument("--spk", default="0,1,2", help="speaker id-jevi razdvojeni zarezom (ignorise se uz --val)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="ignorise se uz --val")
    ap.add_argument("--val", type=int, default=0, help="N recenica iz validacionog skupa + originalni snimci")
    ap.add_argument("--val_filelist", default="./filelists/vits2_val_filelist.txt",
                    help="SIROVI (nefonemizovani) validacioni filelist")
    ap.add_argument("--noise", type=float, default=0.667, help="noise_scale (manje = stabilnije)")
    ap.add_argument("--length", type=float, default=1.0, help="length_scale (>1 sporije, <1 brze)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="./ft_out")
    ap.add_argument("--cpu", action="store_true", help="koristi CPU (ako GPU zauzima trening)")
    args = ap.parse_args()

    hps = utils.get_hparams_from_file(args.config)
    max_ft = hps.train.max_steps if "max_steps" in hps.train.keys() else None

    all_ckpts = sorted(glob.glob(os.path.join(args.logs, "G_*.pth")), key=ckpt_step)
    if not all_ckpts:
        print(f"Nema G_*.pth u {args.logs}")
        sys.exit(1)

    if args.ckpt == "latest":
        # najnoviji po VREMENU IZMENE (u folderu moze da stoji i polazni G_400000 sa vecim brojem)
        ckpts = [max(all_ckpts, key=os.path.getmtime)]
    elif args.ckpt == "all":
        ckpts = [c for c in all_ckpts if max_ft is None or ckpt_step(c) <= max_ft]
        for c in all_ckpts:
            if c not in ckpts:
                print(f"[preskacem] {os.path.basename(c)} (iznad max_steps={max_ft}, verovatno polazni model)")
    else:
        wanted = os.path.join(args.logs, f"G_{int(args.ckpt)}.pth")
        if not os.path.isfile(wanted):
            print(f"Ne postoji {wanted}. Dostupni: {[ckpt_step(c) for c in all_ckpts]}")
            sys.exit(1)
        ckpts = [wanted]

    os.makedirs(args.out, exist_ok=True)

    # poslovi: (oznaka, speaker_id, tekst, putanja_originala ili None)
    jobs = []
    if args.val > 0:
        if not os.path.isfile(args.val_filelist):
            print(f"Ne postoji validacioni filelist: {args.val_filelist}")
            sys.exit(1)
        for i, (wav_path, sid, text) in enumerate(load_val_items(args.val_filelist, args.val)):
            jobs.append((f"val{i}", sid, text, wav_path))
    else:
        for s in args.spk.split(","):
            if s.strip() != "":
                jobs.append(("txt", int(s), args.text, None))

    device = "cpu" if args.cpu else "cuda"

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):
        posterior_channels = 80
        hps.data.use_mel_posterior_encoder = True
    else:
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

    encoded = [get_text(text, hps) for (_, _, text, _) in jobs]

    # originalni snimci i tekstovi (jednom, uz --val)
    for (label, spk, text, wav_path) in jobs:
        if wav_path is None:
            continue
        dst_wav = os.path.join(args.out, f"{label}_original_spk{spk}.wav")
        if os.path.isfile(wav_path):
            if not os.path.isfile(dst_wav):
                shutil.copy(wav_path, dst_wav)
        else:
            print(f"[upozorenje] originalni snimak ne postoji: {wav_path}")
        with open(os.path.join(args.out, f"{label}_spk{spk}_tekst.txt"), "w", encoding="utf-8") as tf:
            tf.write(text + "\n")

    for ckpt in ckpts:
        step = ckpt_step(ckpt)
        _ = utils.load_checkpoint(ckpt, net_g, None)
        for (label, spk, text, _), stn_tst in zip(jobs, encoded):
            torch.manual_seed(args.seed)
            with torch.no_grad():
                x_tst = stn_tst.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([spk]).to(device)
                audio = (
                    net_g.infer(
                        x_tst,
                        x_tst_lengths,
                        sid=sid,
                        noise_scale=args.noise,
                        noise_scale_w=0.0,
                        length_scale=args.length,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
            out_path = os.path.join(args.out, f"ft{step}_{label}_spk{spk}.wav")
            write(data=audio, rate=hps.data.sampling_rate, filename=out_path)
            print(f"Sacuvano: {out_path}")

    print("\nGotovo.")


if __name__ == "__main__":
    main()
