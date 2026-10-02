"""
Konvertuje Piper-stil metadata.csv (format: relpath|speaker_ime|tekst)
u VITS2 multispeaker filelist format (format: puna_putanja.wav|speaker_id|tekst),
sa stratifikovanom train/val/test podelom (90/5/5) po govorniku.

PODESI OVE PROMENLJIVE PRE POKRETANJA:
"""

# ============ PODESI OVDE ============
METADATA_CSV = r"C:\VITS2\dataset\metadata.csv"   # putanja do tvog postojeceg metadata.csv
WAV_ROOT = r"C:\VITS2\dataset\wav"                # folder koji sadrzi speaker0/, speaker1/, speaker2/
OUTPUT_DIR = r"C:\VITS2\dataset"                  # gde da upise nove filelist fajlove
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05  # mora TRAIN_RATIO + VAL_RATIO + TEST_RATIO = 1.0
SEED = 42  # za reproduktivnost podele
# =======================================

import os
import random
import re
from collections import defaultdict

def extract_speaker_id(speaker_name: str) -> int:
    """Izvlaci broj iz 'speaker2' -> 2. Prilagodi ako je format drugaciji."""
    match = re.search(r"(\d+)$", speaker_name)
    if not match:
        raise ValueError(f"Ne mogu da izvucem speaker ID iz: '{speaker_name}'")
    return int(match.group(1))


def main():
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO mora biti 1.0"

    if not os.path.isfile(METADATA_CSV):
        raise FileNotFoundError(f"Ne postoji: {METADATA_CSV}")

    random.seed(SEED)

    # Ucitaj i parsiraj metadata.csv
    entries = []  # (full_wav_path, speaker_id, text)
    skipped = 0
    with open(METADATA_CSV, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"[UPOZORENJE] Linija {line_num} nema 3 dela (preskacem): {line}")
                skipped += 1
                continue
            rel_path, speaker_name, text = parts
            try:
                sid = extract_speaker_id(speaker_name)
            except ValueError as e:
                print(f"[UPOZORENJE] Linija {line_num}: {e} (preskacem)")
                skipped += 1
                continue

            # rel_path je npr. "speaker2/a97" -> treba da postane puna putanja do .wav
            # rel_path vec pocinje sa "speakerX/...", pa ga spajamo sa WAV_ROOT roditeljem
            wav_root_parent = os.path.dirname(WAV_ROOT.rstrip("\\/"))
            full_path = os.path.join(wav_root_parent, os.path.basename(WAV_ROOT), rel_path)
            if not full_path.lower().endswith(".wav"):
                full_path += ".wav"

            if not os.path.isfile(full_path):
                print(f"[UPOZORENJE] Fajl ne postoji, preskacem: {full_path}")
                skipped += 1
                continue

            entries.append((full_path, sid, text))

    if not entries:
        raise RuntimeError("Nijedna linija nije uspesno parsirana. Proveri METADATA_CSV i WAV_ROOT putanje.")

    print(f"\nUcitano {len(entries)} validnih linija ({skipped} preskoceno).")

    # Grupisi po govorniku za stratifikovanu podelu
    by_speaker = defaultdict(list)
    for entry in entries:
        by_speaker[entry[1]].append(entry)

    train_set, val_set, test_set = [], [], []

    print("\nRaspodela po govorniku:")
    for sid in sorted(by_speaker.keys()):
        speaker_entries = by_speaker[sid]
        random.shuffle(speaker_entries)
        n = len(speaker_entries)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # ostatak ide u test da ne izgubimo redove zbog zaokruzivanja
        train_set.extend(speaker_entries[:n_train])
        val_set.extend(speaker_entries[n_train:n_train + n_val])
        test_set.extend(speaker_entries[n_train + n_val:])
        print(f"  speaker{sid}: {n} ukupno -> train={n_train}, "
              f"val={n_val}, test={n - n_train - n_val}")

    # Promesaj svaki split da redosled govornika ne bude blokovski
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def write_filelist(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for full_path, sid, text in data:
                f.write(f"{full_path}|{sid}|{text}\n")

    train_path = os.path.join(OUTPUT_DIR, "vits2_train_filelist.txt")
    val_path = os.path.join(OUTPUT_DIR, "vits2_val_filelist.txt")
    test_path = os.path.join(OUTPUT_DIR, "vits2_test_filelist.txt")

    write_filelist(train_path, train_set)
    write_filelist(val_path, val_set)
    write_filelist(test_path, test_set)

    print(f"\nGotovo:")
    print(f"  train: {len(train_set)} linija -> {train_path}")
    print(f"  val:   {len(val_set)} linija -> {val_path}")
    print(f"  test:  {len(test_set)} linija -> {test_path}")
    print(f"\nBroj govornika (n_speakers za config): {len(by_speaker)}")
    print(f"Speaker ID opseg: {min(by_speaker.keys())} - {max(by_speaker.keys())}")


if __name__ == "__main__":
    main()
