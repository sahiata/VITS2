r"""
preprocess_no_torch_v2.py
------------------------
Ispravljena verzija - pise izlaz INKREMENTALNO (red po red, sa flush) i
ima CHECKPOINT za nastavak posle prekida, umesto da sve drzi u memoriji
i pise tek na kraju (sto je proslu verziju kostalo 3 sata izgubljenog
rada kad se terminal srusio).

Kad naidje na Application Control gresku, ODMAH cuva napredak i
zaustavlja se (isti princip kao karantin kod CUDA gresaka ranije) -
pokreni ponovo, nastavice tacno odatle.

Pokretanje (iz C:\VITS2\vits2_pytorch, sa aktivnim venv):
    python preprocess_no_torch_v2.py --text_index 2 --filelist "C:\ParlaSpeech-RS\train_filelist.txt" --text_cleaners serbian_cleaners
    python preprocess_no_torch_v2.py --text_index 2 --filelist "C:\ParlaSpeech-RS\val_filelist.txt" --text_cleaners serbian_cleaners

(Pokreni JEDNOM PO FAJLU, ne oba odjednom kao ranije - lakse je pratiti
napredak i checkpoint posebno za svaki.)
"""

import argparse
import os
import sys
from text.cleaners import serbian_cleaners

CLEANER_FUNCS = {
    "serbian_cleaners": serbian_cleaners,
}


def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def clean_text(text, cleaner_names):
    for name in cleaner_names:
        text = CLEANER_FUNCS[name](text)
    return text


def read_checkpoint(path):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip() or 0)
        except (ValueError, OSError):
            return 0
    return 0


def write_checkpoint(path, n):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=2, type=int)
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--text_cleaners", nargs="+", default=["serbian_cleaners"])
    args = parser.parse_args()

    in_path = args.filelist
    out_path = in_path + "." + args.out_extension
    checkpoint_path = in_path + ".checkpoint"

    lines = load_lines(in_path)
    total = len(lines)

    already_done = read_checkpoint(checkpoint_path)
    print(f"Ukupno redova: {total}")
    if already_done:
        print(f"Nastavljam od reda {already_done} (checkpoint pronadjen).")

    mode = "a" if already_done > 0 and os.path.isfile(out_path) else "w"
    out_f = open(out_path, mode, encoding="utf-8")

    n_ok = 0
    n_errors = 0

    for i in range(already_done, total):
        parts = lines[i].split("|")
        if len(parts) <= args.text_index:
            print(f"  [upozorenje] red {i+1} nema dovoljno kolona, preskacem")
            write_checkpoint(checkpoint_path, i + 1)
            continue

        original_text = parts[args.text_index]

        try:
            cleaned = clean_text(original_text, args.text_cleaners)
            parts[args.text_index] = cleaned
            out_f.write("|".join(parts) + "\n")
            n_ok += 1
        except Exception as e:
            error_text = str(e)
            if "Application Control" in error_text or "WinError 4551" in error_text:
                out_f.flush()
                out_f.close()
                write_checkpoint(checkpoint_path, i)  # NE i+1 - ovaj red probaj ponovo
                print(f"\nApplication Control blokada na redu {i+1}.")
                print(f"Sacuvano do sad: {n_ok} uspesnih redova.")
                print("Pokreni skriptu PONOVO (ista komanda) da nastavis sa svezim procesom.")
                sys.exit(1)
            else:
                n_errors += 1
                parts[args.text_index] = original_text  # fallback: original tekst
                out_f.write("|".join(parts) + "\n")
                print(f"  [GRESKA] red {i+1}: {original_text[:60]!r} -> {e}")

        if (i + 1) % 500 == 0:
            out_f.flush()
            write_checkpoint(checkpoint_path, i + 1)
            print(f"  ...obradjeno {i+1}/{total}")

    out_f.close()
    write_checkpoint(checkpoint_path, total)

    print(f"\nGotovo. {n_ok} uspesno, {n_errors} gresaka (fallback na original) -> {out_path}")
