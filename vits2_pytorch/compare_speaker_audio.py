## VCTK
import sys
import types

if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None
    sys.modules["segments"] = _fake_segments

import torch
import numpy as np
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    return torch.LongTensor(text_norm)


CONFIG_PATH = "./configs/vits2_srpski.json"
MODEL_PATH = "./logs/srpski_multispeaker_143/G_160000.pth"
TEXT = "Dakle, kada biste ozbiljno prišli ovom problemu, stavili biste i ovo na dnevni red."
SPK_IDS_TO_TEST = [0, 1, 2]

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

stn_tst = get_text(TEXT, hps)

results = {}
SEED = 1234  # ISTI seed za sve, da razlika dodje SAMO od speaker_id, ne od nasumicnosti

for spk_id in SPK_IDS_TO_TEST:
    torch.manual_seed(SEED)  # isti "noise_scale" nasumican seed za svaki poziv
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([spk_id]).cuda()
        audio = (
            net_g.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.0,
                length_scale=1,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
    results[spk_id] = audio
    print(f"speaker {spk_id}: dužina={len(audio)} semplova, mean={audio.mean():.6f}, std={audio.std():.6f}")

print("\n--- Poredjenje parova (isti tekst, isti seed, RAZLICIT speaker_id) ---")
ids = list(results.keys())
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        i, j = ids[a], ids[b]
        audio_i, audio_j = results[i], results[j]
        min_len = min(len(audio_i), len(audio_j))
        diff = audio_i[:min_len] - audio_j[:min_len]
        max_abs_diff = np.abs(diff).max()
        mean_abs_diff = np.abs(diff).mean()
        is_identical = np.array_equal(audio_i, audio_j)
        print(f"speaker {i} vs speaker {j}: IDENTICNO={is_identical}  max|razlika|={max_abs_diff:.6f}  prosek|razlika|={mean_abs_diff:.6f}")
