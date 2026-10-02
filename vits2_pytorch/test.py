import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

CONFIG_PATH = "./configs/vits2_srpski.json"
MODEL_PATH = "./logs/srpski_multispeaker/G_61000.pth"  # ili najnoviji koji imaš
SPK_ID = 0
OUTPUT_WAV_PATH = "test_direct_phonemes.wav"

# Ovo je VEĆ fonemizovan tekst iz training filelist-a (red 15) — koristi
# text_cleaners=[] jer ne treba dodatna fonemizacija
TEXT = "vrˈatɪʃɛ sˈɛ ʊ kˈutɕʊ."

hps = utils.get_hparams_from_file(CONFIG_PATH)
hps.data.use_mel_posterior_encoder = True

net_g = SynthesizerTrn(
    len(symbols),
    80,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
).cuda()
_ = net_g.eval()
_ = utils.load_checkpoint(MODEL_PATH, net_g, None)

# text_cleaners=[] jer je TEXT već fonemizovan, ne treba serbian_cleaners
text_norm = text_to_sequence(TEXT, [])
stn_tst = torch.LongTensor(text_norm)

with torch.no_grad():
    x_tst = stn_tst.cuda().unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    sid = torch.LongTensor([SPK_ID]).cuda()
    audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

write(data=audio, rate=hps.data.sampling_rate, filename=OUTPUT_WAV_PATH)