import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from datetime import datetime

MODEL_NAME = 'melody'
DURATION = 8
CFG_COEF = 3
SAMPLES = 5
PROMPT = 'love pop song with violin, piano arrangement, creating a romantic atmosphere'

melody_conditioned = False

model = MusicGen.get_pretrained(MODEL_NAME)


model.set_generation_params(duration=DURATION,
                            cfg_coef=CFG_COEF)  # generate 8 seconds.
# wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = [PROMPT] * SAMPLES
                # 'A slow and heartbreaking love song at tempo of 60',
                # 'A slow and heartbreaking love song with cello instrument']

if not melody_conditioned:
    wav = model.generate(descriptions, progress=True)  # generates 3 samples.
else:
    melody, sr = torchaudio.load('/home/intern-2023-02/melodytalk/assets/1625.wav')
    wav = model.generate_with_chroma(descriptions, melody[None].expand(SAMPLES, -1, -1), sr, progress=True)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    audio_write(f'output/{current_time}_{idx}',
                one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)