## VCTK
import sys
import time
import types

# Zaobilazimo Application Control blokadu na 'regex' paketu - to je zavisnost
# SegmentsBackend-a unutar phonemizer-a, koji uopste ne koristimo (koristimo
# EspeakBackend preko serbian_cleaners). Stub sprecava da se taj lanac uvoza
# uopste pokusa.
if "segments" not in sys.modules:
    _fake_segments = types.ModuleType("segments")
    _fake_segments.__getattr__ = lambda name: None  # bilo koji atribut vraca None
    sys.modules["segments"] = _fake_segments

import torch

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


CONFIG_PATH = "./logs/srpski_3spk_from143/config.json"
MODEL_PATH = "./logs/srpski_3spk_from143/G_100000.pth"
with open("tekst1.txt", encoding="utf-8") as f:
    TEXT = f.read().strip()
#TEXT = "Dakle, kada biste ozbiljno prišli ovom problemu, stavili biste i ovo na dnevni red. Kuda idemo"
SPK_ID = 1   # 0, 1 ili 2 - tvoji speaker0, speaker1, speaker2
OUTPUT_WAV_PATH = "test_3speakers_100000_spk1.wav"

hps = utils.get_hparams_from_file(CONFIG_PATH)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
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
with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([SPK_ID]).cuda()

    torch.cuda.synchronize()  # da ne meri vreme dok GPU jos zavrsava prethodne poslove
    t_start = time.perf_counter()

    audio = (
        net_g.infer(
            x_tst,
            x_tst_lengths,
            sid=sid,
            noise_scale=0.667,
            noise_scale_w=0.0,
            length_scale=1.15,
        )[0][0, 0]
        .data.cpu()
        .float()
        .numpy()
    )

    torch.cuda.synchronize()  # da sacekamo da GPU STVARNO zavrsi generisanje
    t_end = time.perf_counter()

write(data=audio, rate=hps.data.sampling_rate, filename=OUTPUT_WAV_PATH)

generation_time = t_end - t_start
audio_duration = len(audio) / hps.data.sampling_rate
rtf = generation_time / audio_duration

print(f"\nSacuvano: {OUTPUT_WAV_PATH}")
print(f"Trajanje audio zapisa: {audio_duration:.3f} s")
print(f"Vreme generisanja:     {generation_time:.3f} s")
print(f"RTF (real-time factor): {rtf:.4f}  ({'brze' if rtf < 1 else 'sporije'} od realnog vremena, {1/rtf:.1f}x)" if rtf > 0 else "")