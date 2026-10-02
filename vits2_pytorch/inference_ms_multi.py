## VCTK
import sys
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


CONFIG_PATH = "./configs/vits2_srpski.json"
MODEL_PATH = "./logs/srpski_multispeaker_143/G_160000.pth"
TEXT = "Dakle, kada biste ozbiljno prišli ovom problemu, stavili biste i ovo na dnevni red."
SPK_IDS_TO_TEST = [0, 1, 2]  # menjaj po zelji

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

for spk_id in SPK_IDS_TO_TEST:
    output_path = f"test_160000_spk{spk_id}.wav"
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
    write(data=audio, rate=hps.data.sampling_rate, filename=output_path)
    print(f"Sacuvano: {output_path}")

print("\nGotovo. Poslusaj sve fajlove i uporedi.")
