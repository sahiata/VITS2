"""
Test skript: proverava da li serbian_cleaners (eSpeak-ng fonemizacija za srpski)
proizvodi IPA simbole koji NISU u text/symbols.py vokabularu.
Ako ima nepokrivenih simbola, oni bi bili TIHO PRESKOCENI tokom pravog treninga
(vidi cleaned_text_to_sequence u text/__init__.py) - ovaj skript to otkriva unapred.

POKRENI IZ ROOT FOLDERA REPO-A (C:\\VITS2\\vits2_pytorch), NE iz text/ foldera,
jer se oslanja na 'text' i 'text.cleaners' module kao u ostatku projekta.
"""

from text import cleaners
from text.symbols import symbols

symbol_set = set(symbols)

# Testni srpski recenice - namerno biraj recenice koje pokrivaju sve "problematicne" glasove
test_sentences = [
    "Čučeći ćuška ždrebe žuto, lišće šušti u šumi.",
    "Konj njiska, a ljubav lije kao đubre u đaka.",
    "Džak pun džemova drži čovek pored kuće.",
    "Pretpostavljam da će stići kasnije, ali nisam sigurna.",
]

print(f"Ukupno simbola u vokabularu: {len(symbol_set)}")
print(f"Vokabular: {sorted(symbol_set)}\n")

all_missing = set()

for sentence in test_sentences:
    phonemes = cleaners.serbian_cleaners(sentence)
    missing = set(phonemes) - symbol_set
    status = "OK" if not missing else f"NEDOSTAJE: {sorted(missing)}"
    print(f"Recenica: {sentence}")
    print(f"Fonemi:   {phonemes}")
    print(f"Status:   {status}\n")
    all_missing |= missing

print("=" * 60)
if all_missing:
    print(f"UPOZORENJE: {len(all_missing)} simbol(a) van vokabulara: {sorted(all_missing)}")
    print("Ovi karakteri bi bili TIHO PRESKOCENI tokom treninga.")
else:
    print("Svi fonemi iz test recenica su pokriveni vokabularom. Bezbedno za preprocessing.")