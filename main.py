import os

import torchaudio
from melodytalk.audiocraft.models import MusicGen
from melodytalk.audiocraft.data.audio import audio_write
from datetime import datetime
import torch

MODEL_NAME = 'melody'
DURATION = 40
CFG_COEF = 3
SAMPLES = 10
# PROMPT = 'music loop. Passionate love song with guitar rhythms, electric piano chords, drums pattern. instrument: guitar, piano, drum.'
PROMPT = "rock music loop with rhythmic, quick and technic saxophone solo. bpm: 90. instrument: saxophone, guitar, drum."
melody_conditioned = True

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model = MusicGen.get_pretrained(MODEL_NAME, device='cuda')

DURATION_1 = min(DURATION, 30)
DURATION_2 = max(DURATION - 30, 0)
OVERLAP = 8

model.set_generation_params(duration=DURATION_1,
                            cfg_coef=CFG_COEF,)  # generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = [PROMPT] * SAMPLES
                # 'A slow and heartbreaking love song at tempo of 60',
                # 'A slow and heartbreaking love song with cello instrument']

def generate():
    if not melody_conditioned:
        wav = model.generate(descriptions, progress=True)  # generates 3 samples.
    else:
        melody, sr = torchaudio.load('/home/intern-2023-02/melodytalk/assets/20230705-155518_3.wav')
        wav = model.generate_continuation(melody[None].expand(SAMPLES, -1, -1), sr, descriptions, progress=True)
        if DURATION_2 > 0:
            wav_ = wav[:, :, -OVERLAP * model.sample_rate:]
            model.set_generation_params(duration=(OVERLAP + DURATION_2))
            wav_2 = model.generate_continuation(wav_, model.sample_rate, descriptions, progress=True)[..., OVERLAP * model.sample_rate:]
            wav = torch.cat([wav, wav_2], dim=-1)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        audio_write(f'output/{current_time}_{idx}',
                    one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

generate()