r"""
Izvlači samo jednog govornika iz Piper-formata metadata fajla
(putanja|speaker_id|tekst) i pravi VITS2 filelist (putanja|tekst),
sa train/val split-om.

Pokretanje (primer):
    python extract_speaker_vits2.py --metadata C:\VITS2\dataset\metadata --speaker speaker1 --wav-root C:\VITS2\dataset\wav --val-ratio 0.02

Izlaz (u istom folderu odakle se pokrece skripta, ili --out-dir ako je zadat):
    speaker1_audio_text_train_filelist.txt
    speaker1_audio_text_val_filelist.txt
"""

import argparse
import os
import random


def parse_metadata(metadata_path, target_speaker):
    """
    Ocekivani format po liniji: path|speaker_id|text
    npr:  speaker0/a10|speaker0|Nisam imao vremena...
    Vraca listu (wav_path, text) samo za target_speaker.
    """
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 3:
                print(f"[upozorenje] preskačem liniju {line_num} (neočekivan format): {line[:80]}")
                continue
            rel_path, speaker_id, text = parts[0], parts[1], "|".join(parts[2:])
            if speaker_id != target_speaker:
                continue
            rows.append((rel_path, text))
    return rows


def build_wav_path(rel_path, wav_root, ext=".wav"):
    """
    rel_path je oblika 'speaker1/a10' (bez ekstenzije).
    Spaja sa wav_root i dodaje ekstenziju ako fali.
    """
    if not rel_path.lower().endswith(ext):
        rel_path = rel_path + ext
    # normalize slashes for Windows
    rel_path = rel_path.replace("/", os.sep)
    full_path = os.path.join(wav_root, rel_path) if wav_root else rel_path
    return full_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=r"C:\VITS2\dataset\metadata.csv", help="Putanja do metadata (Piper CSV) fajla")
    ap.add_argument("--speaker", default="speaker1", help="Speaker ID koji se izvlači, npr. speaker1")
    ap.add_argument("--wav-root", default=r"C:\VITS2\dataset\wav", help="Root folder wav fajlova.")
    ap.add_argument("--val-ratio", type=float, default=0.02, help="Udeo za validaciju (default 2%)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", default=r"C:\VITS2\vits2_pytorch\filelists", help="Gde da se sačuvaju izlazni filelist-ovi")
    ap.add_argument("--check-files", action="store_true", default=True,
                     help="Proveri da li svaki wav fajl zaista postoji na disku (upozorenje ako ne)")
    args = ap.parse_args()

    rows = parse_metadata(args.metadata, args.speaker)
    if not rows:
        print(f"Nije pronađen nijedan red za speaker='{args.speaker}'. Proveri ID u metadata fajlu.")
        return

    print(f"Pronađeno {len(rows)} redova za '{args.speaker}'.")

    entries = []
    missing = 0
    for rel_path, text in rows:
        wav_path = build_wav_path(rel_path, args.wav_root)
        if args.check_files and not os.path.isfile(wav_path):
            missing += 1
        entries.append((wav_path, text))

    if args.check_files and missing:
        print(f"[upozorenje] {missing} wav fajlova nije pronađeno na disku (proveri --wav-root).")

    random.seed(args.seed)
    random.shuffle(entries)

    n_val = max(1, int(len(entries) * args.val_ratio))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, f"{args.speaker}_audio_text_train_filelist.txt")
    val_path = os.path.join(args.out_dir, f"{args.speaker}_audio_text_val_filelist.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for wav_path, text in train_entries:
            f.write(f"{wav_path}|{text}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for wav_path, text in val_entries:
            f.write(f"{wav_path}|{text}\n")

    print(f"Train: {len(train_entries)} redova -> {train_path}")
    print(f"Val:   {len(val_entries)} redova -> {val_path}")
    print("\nNapomena: ovi fajlovi sadrže SIROV tekst (nisu 'cleaned').")
    print("Ako tvoj VITS2 fork zahteva .cleaned filelist (kao u config.json: "
          "'...train_filelist.txt.cleaned'), pokreni preprocess/phonemize skriptu "
          "iz repo-a nad ovim fajlovima da dobiješ finalnu .cleaned verziju sa srpskim cleaner-om.")


if __name__ == "__main__":
    main()