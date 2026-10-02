## VCTK
import sys
import types

if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None
    sys.modules["segments"] = _fake_segments

import torch
import utils
from models import SynthesizerTrn
from text.symbols import symbols

CONFIG_PATH = "./configs/vits2_srpski.json"
MODEL_PATH = "./logs/srpski_multispeaker_143/G_160000.pth"

hps = utils.get_hparams_from_file(CONFIG_PATH)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    posterior_channels = 80
    hps.data.use_mel_posterior_encoder = True
else:
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).cuda()
_ = net_g.eval()
_ = utils.load_checkpoint(MODEL_PATH, net_g, None)

print(f"n_speakers u configu: {hps.data.n_speakers}")
print(f"Da li model ima emb_g (speaker embedding tabelu)? {hasattr(net_g, 'emb_g')}")

if hasattr(net_g, "emb_g") and net_g.emb_g is not None:
    emb_table = net_g.emb_g.weight.data  # oblik: (n_speakers, gin_channels)
    print(f"Oblik embedding tabele: {emb_table.shape}")

    ids_to_check = [0, 1, 2, 10, 50, 100]
    vectors = {i: emb_table[i].cpu() for i in ids_to_check if i < emb_table.shape[0]}

    print("\nProsecna apsolutna vrednost svakog vektora (da vidimo da nisu svi nule):")
    for i, v in vectors.items():
        print(f"  speaker {i}: mean|v|={v.abs().mean().item():.6f}  std={v.std().item():.6f}")

    print("\nCosine similarity izmedju parova (1.0 = identicno, 0.0 = potpuno razlicito):")
    ids_list = list(vectors.keys())
    for a in range(len(ids_list)):
        for b in range(a + 1, len(ids_list)):
            i, j = ids_list[a], ids_list[b]
            v1, v2 = vectors[i], vectors[j]
            cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            print(f"  speaker {i} vs speaker {j}: {cos_sim:.4f}")
else:
    print("\nPROBLEM: model NEMA emb_g tabelu - to bi objasnilo zasto svi zvuce isto!")
    print("Proveri da li je 'use_spk_conditioned_encoder': true i 'gin_channels' u configu.")
