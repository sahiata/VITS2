## VCTK
import re
import sys
import types

# isti trik kao ranije (zaobilazi Application Control blokadu na 'regex')
if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None
    sys.modules["segments"] = _fake_segments

from text.cleaners import serbian_cleaners

# ---------------------------------------------------------------
# 1) Direktan test: isti tekst sa '+' i bez '+'
# ---------------------------------------------------------------
EXAMPLES = [
    "p+otpuno obrnuli",
    "potpuno obrnuli",
    "D+akle, k+ada b+iste +ozbiljno",
    "Dakle, kada biste ozbiljno",
]

print("=== 1) Fonemizacija sa i bez '+' ===")
for s in EXAMPLES:
    print(f"{s!r}\n   -> {serbian_cleaners(s)}\n")

# ---------------------------------------------------------------
# 2) Provera na pravim podacima: sirov filelist vs .cleaned
# ---------------------------------------------------------------
RAW = r"C:\ParlaSpeech-RS\train_filelist.txt"
CLEANED = r"C:\ParlaSpeech-RS\train_filelist.txt.cleaned"
MAX_LINES = 5000

PLUS_PHONEMES = re.compile(r"pl[ˈˌ]?[uʊ]s")

with open(RAW, encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f if l.strip()]
with open(CLEANED, encoding="utf-8") as f:
    clean_lines = [l.rstrip("\n") for l in f if l.strip()]

lines_with_plus = 0
total_plus_chars = 0
matches_in_plus_lines = 0
lines_without_plus = 0
matches_in_plain_lines = 0
examples_shown = 0

for raw, clean in list(zip(raw_lines, clean_lines))[:MAX_LINES]:
    rp, cp = raw.split("|"), clean.split("|")
    if len(rp) < 3 or len(cp) < 3:
        continue
    raw_text, clean_text = rp[2], cp[2]
    n_plus = raw_text.count("+")
    n_match = len(PLUS_PHONEMES.findall(clean_text))
    if n_plus > 0:
        lines_with_plus += 1
        total_plus_chars += n_plus
        matches_in_plus_lines += n_match
        if examples_shown < 3:
            print(f"PRIMER: {raw_text}\n   -> {clean_text}\n")
            examples_shown += 1
    else:
        lines_without_plus += 1
        matches_in_plain_lines += n_match

print("=== 2) Statistika (prvih %d redova) ===" % MAX_LINES)
print(f"Redova SA '+' u tekstu: {lines_with_plus}")
print(f"  ukupno '+' znakova: {total_plus_chars}")
print(f"  'plus' fonema u njihovim fonemizovanim redovima: {matches_in_plus_lines}")
if total_plus_chars:
    print(f"  odnos (plus fonema / '+' znakova): {matches_in_plus_lines / total_plus_chars:.2f}")
print(f"Redova BEZ '+': {lines_without_plus}")
print(f"  'plus' fonema u tim redovima (prirodna reč 'plus', osnova poredjenja): {matches_in_plain_lines}")
