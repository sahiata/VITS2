"""
fix_duplicate_cleaned_lines.py
------------------------
Uklanja duplirane redove iz train_filelist.txt.cleaned (nastale zbog
neuspesnog skracivanja fajla pre poslednjeg nastavka) - zadrzava PRVU
pojavu svakog reda po audio putanji (prva kolona pre '|'), izbacuje
ponovljene.

Pokretanje:
    python fix_duplicate_cleaned_lines.py
"""

path = r"C:\ParlaSpeech-RS\train_filelist.txt.cleaned"

with open(path, "r", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]

print(f"Ucitano redova: {len(lines)}")

seen = set()
deduped = []
duplicates = 0

for line in lines:
    audio_path = line.split("|", 1)[0]
    if audio_path in seen:
        duplicates += 1
        continue
    seen.add(audio_path)
    deduped.append(line)

with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(deduped) + "\n")

print(f"Uklonjeno duplikata: {duplicates}")
print(f"Preostalo redova: {len(deduped)}")
