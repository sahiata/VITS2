r"""
Skenira CEO .cleaned filelist i proverava da li svaki karakter u fonemizovanom
tekstu postoji u text/symbols.py. Ako neki karakter fali, tokenizer moze da
vrati indeks van granica embedding tabele, sto na GPU-u izaziva access
violation (crash bez citljive Python greske) umesto lepe exception poruke.

Pokretanje (iz C:\VITS2\vits2_pytorch, sa aktivnim venv):
    python check_symbols_coverage.py
"""

import os
from text.symbols import symbols

FILELISTS = [
    "filelists/speaker1_audio_text_train_filelist.txt.cleaned",
    "filelists/speaker1_audio_text_val_filelist.txt.cleaned",
]

symbol_set = set(symbols)


def check_file(path):
    if not os.path.isfile(path):
        print(f"[upozorenje] Ne postoji: {path}")
        return

    bad_lines = 0
    all_missing_chars = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue
            text = "|".join(parts[1:])
            missing = [ch for ch in text if ch not in symbol_set]
            if missing:
                bad_lines += 1
                if bad_lines <= 15:
                    print(f"  {path}:{line_num} -> nepoznati karakteri {sorted(set(missing))!r} u: {text[:80]!r}")
                for ch in missing:
                    all_missing_chars[ch] = all_missing_chars.get(ch, 0) + 1

    print(f"\n{path}: {bad_lines} problematičnih linija")
    if all_missing_chars:
        print("Svi nepoznati karakteri i broj pojavljivanja:")
        for ch, count in sorted(all_missing_chars.items(), key=lambda x: -x[1]):
            print(f"    {ch!r}  (U+{ord(ch):04X})  x{count}")
    print()


def main():
    print(f"symbols.py ima {len(symbols)} simbola ukupno.\n")
    for path in FILELISTS:
        check_file(path)


if __name__ == "__main__":
    main()