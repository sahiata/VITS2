## VCTK
"""
rephonemize_without_plus.py
------------------------
Popravlja .cleaned filelist-ove: espeak je '+' citao kao rec "plus" i sekao
reci (npr. 'p+otpuno' -> 'pˈə plˈus ˈotpʊno'). Ova skripta:

  - redove BEZ '+' ostavlja tacno kakvi jesu (vec su ispravni)
  - u redovima SA '+' skida '+' iz teksta pa ih ponovo fonemizuje serbian
    pipeline-om (lowercase -> phonemize -> collapse_whitespace), ali u
    SERIJAMA sa jednim espeak backend-om (mnogo brze nego red po red)
  - stari .cleaned fajl NE brise, nego ga preimenuje u .cleaned.bak_plus
  - ima progress fajl: ako naidje na Application Control gresku, sacuva
    napredak i zaustavi se - samo pokreni ISTU komandu ponovo

Pokretanje (iz C:\\VITS2\\vits2_pytorch, venv aktivan, PHONEMIZER_ESPEAK_LIBRARY postavljen):
    python rephonemize_without_plus.py
"""
import os
import re
import sys
import types

if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None
    sys.modules["segments"] = _fake_segments

from phonemizer import phonemize
from text.cleaners import lowercase, collapse_whitespace

FILELISTS = [
    r"C:\ParlaSpeech-RS\train_filelist.txt",
    r"C:\ParlaSpeech-RS\val_filelist.txt",
]
TEXT_INDEX = 2
CHUNK = 500
PLUS_PHONEMES = re.compile(r"pl[ˈˌ]?[uʊ]s")


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def phonemize_batch(texts):
    texts = [collapse_whitespace(lowercase(t)) for t in texts]
    out = phonemize(
        texts,
        language="sr",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )
    assert len(out) == len(texts), "broj rezultata se ne poklapa sa ulazom"
    return [collapse_whitespace(p) for p in out]


def process(raw_path):
    cleaned_path = raw_path + ".cleaned"
    new_path = cleaned_path + ".new"
    progress_path = new_path + ".progress"
    bak_path = cleaned_path + ".bak_plus"

    raw_lines = read_lines(raw_path)
    old_clean = read_lines(cleaned_path) if os.path.isfile(cleaned_path) else None

    # ako je vec zavrseno u ranijem pokretanju
    if os.path.isfile(bak_path) and not os.path.isfile(new_path):
        print(f"[preskacem] {os.path.basename(raw_path)} - vec popravljeno (postoji {os.path.basename(bak_path)})")
        return

    if old_clean is None or len(old_clean) != len(raw_lines):
        print(f"GRESKA: {cleaned_path} ne postoji ili broj redova ({None if old_clean is None else len(old_clean)}) "
              f"ne odgovara sirovom ({len(raw_lines)}). Zaustavljam.")
        sys.exit(1)

    total = len(raw_lines)
    done = 0
    if os.path.isfile(progress_path) and os.path.isfile(new_path):
        done = int(open(progress_path).read().strip() or 0)
        print(f"Nastavljam {os.path.basename(raw_path)} od reda {done}/{total}")
    else:
        open(new_path, "w", encoding="utf-8").close()

    out_f = open(new_path, "a", encoding="utf-8")
    fixed = 0

    while done < total:
        end = min(done + CHUNK, total)
        chunk_raw = raw_lines[done:end]
        chunk_old = old_clean[done:end]

        # koji redovi u ovoj seriji imaju '+'
        need_idx = []
        need_texts = []
        for k, rline in enumerate(chunk_raw):
            parts = rline.split("|")
            if len(parts) > TEXT_INDEX and "+" in parts[TEXT_INDEX]:
                need_idx.append(k)
                need_texts.append(parts[TEXT_INDEX].replace("+", ""))

        new_phonemes = {}
        if need_texts:
            try:
                results = phonemize_batch(need_texts)
            except Exception as e:
                msg = str(e)
                out_f.flush()
                out_f.close()
                open(progress_path, "w").write(str(done))
                if "Application Control" in msg or "WinError 4551" in msg:
                    print(f"\nApplication Control blokada oko reda {done}. Napredak sacuvan. Pokreni ISTU komandu ponovo.")
                else:
                    print(f"\nGRESKA oko reda {done}: {e}")
                sys.exit(1)
            for k, ph in zip(need_idx, results):
                new_phonemes[k] = ph

        for k in range(len(chunk_raw)):
            old_parts = chunk_old[k].split("|")
            if k in new_phonemes:
                old_parts[TEXT_INDEX] = new_phonemes[k]
                fixed += 1
            out_f.write("|".join(old_parts) + "\n")

        out_f.flush()
        done = end
        open(progress_path, "w").write(str(done))
        print(f"  {os.path.basename(raw_path)}: {done}/{total}  (popravljeno redova u ovoj sesiji: {fixed})")

    out_f.close()

    # provera: ne sme da ostane nijedno 'plus'
    remaining = 0
    for line in read_lines(new_path):
        p = line.split("|")
        if len(p) > TEXT_INDEX and PLUS_PHONEMES.search(p[TEXT_INDEX]):
            remaining += 1
    print(f"Provera: redova koji i dalje sadrze 'plus' fonem: {remaining}")

    os.replace(cleaned_path, bak_path)
    os.replace(new_path, cleaned_path)
    if os.path.isfile(progress_path):
        os.remove(progress_path)
    print(f"Gotovo: {cleaned_path}  (stari sacuvan kao {bak_path})\n")


if __name__ == "__main__":
    for fl in FILELISTS:
        process(fl)
    print("Sve zavrseno.")
