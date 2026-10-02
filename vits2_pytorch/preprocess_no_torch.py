r"""
preprocess_no_torch.py
------------------------
Isto sto i preprocess.py (fonemizuje tekst preko serbian_cleaners), ali
NE uvozi torch - zaobilazi Windows Application Control blokadu na
c10_cuda.dll, koja nema nikakve veze sa ovim zadatkom (fonemizacija je
cist CPU/tekst posao, torch ovde nije ni potreban).

Pokretanje (iz C:\VITS2\vits2_pytorch, sa aktivnim venv):
    python preprocess_no_torch.py --text_index 2 --filelists "C:\ParlaSpeech-RS\train_filelist.txt" "C:\ParlaSpeech-RS\val_filelist.txt" --text_cleaners serbian_cleaners
"""

import argparse
from text.cleaners import serbian_cleaners

CLEANER_FUNCS = {
    "serbian_cleaners": serbian_cleaners,
}


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        lines = [line.strip().split(split) for line in f if line.strip()]
    return lines


def clean_text(text, cleaner_names):
    for name in cleaner_names:
        if name not in CLEANER_FUNCS:
            raise ValueError(f"Nepoznat cleaner: {name}")
        text = CLEANER_FUNCS[name](text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "filelists/ljs_audio_text_val_filelist.txt",
            "filelists/ljs_audio_text_test_filelist.txt",
        ],
    )
    parser.add_argument("--text_cleaners", nargs="+", default=["english_cleaners2"])

    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)
        n_errors = 0
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][args.text_index]
            try:
                cleaned_text = clean_text(original_text, args.text_cleaners)
            except Exception as e:
                print(f"  [GRESKA] red {i+1}: {original_text[:60]!r} -> {e}")
                cleaned_text = original_text
                n_errors += 1
            filepaths_and_text[i][args.text_index] = cleaned_text

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

        print(f"  Zavrseno: {len(filepaths_and_text)} redova, {n_errors} gresaka -> {new_filelist}")
