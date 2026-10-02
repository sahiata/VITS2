r"""
Generise .cleaned filelist-ove iz sirovih VITS2 filelist-ova, propustajuci
tekst kroz serbian_cleaners (phonemizer/espeak-ng) - isto sto se inace desi
"on the fly" tokom treninga ako cleaned_text=False, ali ovde radimo unapred
jer config.json ima "cleaned_text": true.

Pokretanje (iz C:\VITS2\vits2_pytorch, sa aktivnim venv):
    python make_cleaned_filelists.py

Cita:
    filelists/speaker1_audio_text_train_filelist.txt
    filelists/speaker1_audio_text_val_filelist.txt
Pise:
    filelists/speaker1_audio_text_train_filelist.txt.cleaned
    filelists/speaker1_audio_text_val_filelist.txt.cleaned
"""

import os
from text.cleaners import serbian_cleaners

FILELISTS_DIR = "filelists"
INPUTS = [
    "speaker1_audio_text_train_filelist.txt",
    "speaker1_audio_text_val_filelist.txt",
]


def process_file(in_path, out_path):
    n_ok = 0
    n_failed = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                print(f"[upozorenje] {in_path}:{line_num} preskočena (neočekivan format): {line[:80]}")
                n_failed += 1
                continue
            wav_path = parts[0]
            text = "|".join(parts[1:])
            try:
                cleaned = serbian_cleaners(text)
            except Exception as e:
                print(f"[GREŠKA] {in_path}:{line_num} nije mogla da se fonemizuje: {text[:60]!r} -> {e}")
                n_failed += 1
                continue
            fout.write(f"{wav_path}|{cleaned}\n")
            n_ok += 1
    print(f"{in_path} -> {out_path}: {n_ok} OK, {n_failed} preskočeno/grešaka")


def main():
    for fname in INPUTS:
        in_path = os.path.join(FILELISTS_DIR, fname)
        out_path = in_path + ".cleaned"
        if not os.path.isfile(in_path):
            print(f"[upozorenje] Ne postoji: {in_path} — preskačem.")
            continue
        process_file(in_path, out_path)
    print("\nGotovo. Proveri da li config.json 'training_files'/'validation_files' pokazuju na .cleaned fajlove.")


if __name__ == "__main__":
    main()