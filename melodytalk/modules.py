import os
import uuid
import torch
from shutil import copyfile
import torchaudio

# text2music
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
# source separation
import demucs.separate

from utils import prompts, get_new_audio_name, description_to_attributes, cut_dialogue_history


# Initialze common models
musicgen_model = MusicGen.get_pretrained('melody')
musicgen_model.set_generation_params(duration=8)

class Text2Music(object):
    def __init__(self, device):
        print("Initializing Text2Music")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Generate music from user input text",
        description="useful if you want to generate music from a user input text and save it to a file."
                    "like: generate music of love pop song, or generate music with piano and violin."
                    "The input to this tool should be a string, representing the text used to generate music."
    )

    def inference(self, text):
        music_filename = os.path.join("music", f"{str(uuid.uuid4())[:8]}.wav")
        text = description_to_attributes(text)  # convert text to attributes
        wav = self.model.generate([text], progress=False)
        wav = wav[0]  # batch size is 1
        audio_write(music_filename[:-4],
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed Text2Music, Input Text: {text}, Output Music: {music_filename}.")
        return music_filename

class Text2MusicWithMelody(object):
    def __init__(self, device):
        print("Initializing Text2MusicWithMelody")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Generate music from user input text with melody or track condition",
        description="useful if you want to generate, style transfer or remix music from a user input text with a given melody or track condition."
                    "like: remix the given melody with text description, or doing style transfer as text described with the given melody."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the text description."
    )

    def inference(self, inputs):
        music_filename, text = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        text = description_to_attributes(text)  # convert text to attributes
        print(f"Generating music from text with melody condition, Input Text: {text}, Melody: {music_filename}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="remix")
        melody, sr = torchaudio.load(music_filename)
        wav = self.model.generate_with_chroma([text], melody[None].expand(1, -1, -1), sr, progress=False)
        wav = wav[0]  # batch size is 1
        audio_write(updated_music_filename[:-4],
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed Text2MusicWithMelody, Output Music: {updated_music_filename}.")
        return updated_music_filename





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
        name="Separate one track from a music file to extract (return the single track) or remove (return the mixture of the rest tracks) it.",
        description="useful if you want to separate a track (must be one of `vocals`, `drums`, `bass`, `guitar`, `piano` or `other`) from a music file."
                    "Like: separate vocals from a music file, or remove the drum track from a music file."
                    "The input to this tool should be a comma separated string of three params, "
                    "representing the music_filename, the specific track name, and the mode (must be `extract` or `remove`)."
    )

    def inference(self, inputs):
        music_filename, instrument, mode = inputs.split(",")[0].strip(), inputs.split(",")[1].strip(), inputs.split(",")[2].strip()
        print(f"{mode} {instrument} track from {music_filename}.")

        if mode == "extract":
            instrument_mode = instrument
        elif mode == "remove":
            instrument_mode = f"no_{instrument}"
        else:
            raise ValueError("mode must be `extract` or `remove`.")

        updated_music_filename = get_new_audio_name(music_filename, func_name=f"{instrument_mode}")

        # fill params
        self.params_list[-2] = instrument
        self.params_list[-1] = music_filename
        # run
        demucs.separate.main(self.params_list)
        # rename
        copyfile(
            os.path.join("separated", "htdemucs_6s", music_filename[:-4].split("/")[-1],f"{instrument_mode}.wav"),
            updated_music_filename
        )
        # delete the folder
        # os.system(f"rm -rf {os.path.join('separated', 'htdemucs_6s')}")

        print(f"Processed Source Separation, Input Music: {music_filename}, Output Instrument: {instrument_mode}.")
        return updated_music_filename


class SimpleTracksMixing(object):
    def __init__(self, device):
        print("Initializing SimpleTracksMixing")
        self.device = device

    @prompts(
        name="Simply mixing two tracks from two music files.",
        description="useful if you want to mix two tracks from two music files."
                    "Like: mix the vocals track from a music file with the drums track from another music file."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the first music_filename_1 and the second music_filename_2."
    )

    def inference(self, inputs):
        music_filename_1, music_filename_2 = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        print(f"Mixing two tracks from two music files, Input Music 1: {music_filename_1}, Input Music 2: {music_filename_2}.")
        updated_music_filename = get_new_audio_name(music_filename_1, func_name="mixing")
        # load
        wav_1, sr_1 = torchaudio.load(music_filename_1)
        wav_2, sr_2 = torchaudio.load(music_filename_2)
        # resample
        if sr_1 != sr_2:
            wav_2 = torchaudio.transforms.Resample(sr_2, sr_1)(wav_2)
        # pad or cut
        if wav_1.shape[-1] > wav_2.shape[-1]:
            wav_2 = torch.cat([wav_2, torch.zeros_like(wav_1[:, wav_2.shape[-1]:])], dim=-1)
        elif wav_1.shape[-1] < wav_2.shape[-1]:
            wav_2 = wav_2[:, :wav_1.shape[-1]]
        # mix
        assert wav_1.shape == wav_2.shape  # channel, length
        wav = torch.add(wav_1, wav_2)
        # write
        audio_write(updated_music_filename[:-4],
                    wav.cpu(), sr_1, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed TracksMixing, Output Music: {updated_music_filename}.")
        return updated_music_filename