import os
import uuid
import torch
from shutil import copyfile

# text2music
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# source separation
import demucs.separate

from utils import prompts, get_new_audio_name

class Text2Music(object):
    def __init__(self, device):
        print("Initializing Text2Music")
        self.device = device
        self.model = MusicGen.get_pretrained('melody')

        # Set generation params
        self.model.set_generation_params(duration=8)

    @prompts(
        name="Generate music from user input text",
        description="useful if you want to generate music from a user input text and save it to a file."
                    "like: generate music of love pop song, or generate music with piano and violin."
                    "The input to this tool should be a string, representing the text used to generate music."
    )

    def inference(self, text):
        music_filename = os.path.join("music", f"{str(uuid.uuid4())[:8]}.wav")
        prompt = text
        wav = self.model.generate([text], progress=False)
        wav = wav[0]  # batch size is 1
        audio_write(music_filename[:-4],
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed Text2Music, Input Text: {text}, Output Music: {music_filename}.")
        return music_filename


class ExtractTrack(object):
    def __init__(self, device):
        print("Initializing ExtractTrack")
        self.device = device
        self.params_list = [
            "-n", "htdemucs_6s",  # model selection
            "--two-stems", None,  # track name
            None                  # original filename
        ]

    @prompts(
        name="Extract one track from a music file",
        description="useful if you want to separate a track (must be one of `vocals`, `drums`, `bass`, `guitar`, `piano` or `other`) from a music file."
                    "Like: separate vocals from a music file, or extract drums from a music file."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the specific track name."
    )

    def inference(self, inputs):
        music_filename, instrument = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        print(f"Extracting {instrument} track from {music_filename}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name=f"{instrument}")

        # fill params
        self.params_list[-2] = instrument
        self.params_list[-1] = music_filename
        # run
        demucs.separate.main(self.params_list)
        # rename
        copyfile(
            os.path.join("separated", "htdemucs_6s", music_filename[:-4].split("/")[-1],f"{instrument}.wav"),
            updated_music_filename
        )
        # delete the folder
        # os.system(f"rm -rf {os.path.join('separated', 'htdemucs_6s')}")

        print(f"Processed Source Separation, Input Music: {music_filename}, Output Instrument: {instrument}.")
        return updated_music_filename
