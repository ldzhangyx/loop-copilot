from shutil import copyfile
from dataclasses import dataclass

import librosa
import torch
import torchaudio.functional
import pedalboard

# text2music
from melodytalk.dependencies.audiocraft.models import MusicGen
from melodytalk.dependencies.audiocraft.data.audio import audio_write
# source separation
import demucs.separate
# CLAP
from melodytalk.dependencies import laion_clap
# Vampnet
from melodytalk.dependencies.vampnet.interface import Interface
from melodytalk.dependencies.vampnet.main import vamp
# captioning
from melodytalk.dependencies.lpmc.music_captioning.captioning import main as captioning

from utils import *

DURATION = 8
GENERATION_CANDIDATE = 6

# Initialze common models
# musicgen_model = MusicGen.get_pretrained('large')
# musicgen_model.set_generation_params(duration=DURATION)

musicgen_model = MusicGen.get_pretrained('melody')
musicgen_model.set_generation_params(duration=DURATION)

# musicgen_model = torch.compile(musicgen_model)

# Intialize CLIP post filter
CLAP_model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device="cuda")
CLAP_model.load_ckpt("/home/intern-2023-02/melodytalk/melodytalk/pretrained/music_audioset_epoch_15_esc_90.14.pt")

# Vampnet
interface = Interface(
    coarse_ckpt="./dependencies/vampnet/models/vampnet/coarse.pth",
    coarse2fine_ckpt="./dependencies/vampnet/models/vampnet/c2f.pth",
    codec_ckpt="./dependencies/vampnet/models/vampnet/codec.pth",
    wavebeat_ckpt="./dependencies/vampnet/models/wavebeat.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# captioning model


@dataclass
class GlobalAttributes(object):
    # metadata
    key: str = None
    bpm: int = None
    # genre: str = None
    # mood: str = None
    instrument: tp.List[str] = None
    # text description cache
    descriptions: str = None
    # tracks cache
    mix: torch.Tensor = None
    stems: tp.Dict[str, torch.Tensor] = None

    def __post_init__(self):
        self.instrument = []
        self.stems = {}

    def update_attributes_from_description(self, description=None):
        attributes_list = description.split('.')[1:]
        for i, attribute in enumerate(attributes_list):
            if "key:" in attribute:
                self.key = attribute.split('key:')[-1].strip()
            elif "bpm:" in attribute:
                self.bpm = int(attribute.split('bpm:')[-1].strip())
            elif "instrument:" in attribute:
                self.instrument = attribute.split('instrument:')[-1].strip().split(',')

    def description_to_attributes_wrapper(self, description: str) -> str:
        formatted_description = description_to_attributes(description)
        self.descriptions = formatted_description
        if any([x in formatted_description for x in ["key:", "bpm:", "instrument:"]]):
            self.update_attributes_from_description(formatted_description)
        return formatted_description




# attribute management
attribute_table = GlobalAttributes()

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
        attribute_table.descriptions = text
        text = description_to_attributes(text)  # convert text to attributes
        wav = self.model.generate([text], progress=False)
        wav = wav[0]  # batch size is 1
        audio_write(music_filename[:-4],
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed Text2Music, Input Text: {text}, Output Music: {music_filename}.")
        return music_filename

class Text2MusicWithTitle(object):
    def __init__(self, device):
        print("Initializing Text2MusicWithTitle")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Generate music from user input when the input is a title of music",
        description="useful if you want to generate music which feels like an real music."
                    "like: generate music that feels like 'hey jude', or generate music similar to 'let it be'."
                    "The input to this tool should be a string, "
                    "representing the music title."
    )

    def inference(self, inputs):
        title = inputs
        music_filename = os.path.join("music", f"{title}.wav")
        text = music_title_to_description(title)  # using chatGPT's knowledge base to convert title to description
        attribute_table.descriptions = text
        text = description_to_attributes(text)  # convert text to attributes
        wav = self.model.generate([text], progress=False)
        wav = wav[0]  # batch size is 1
        audio_write(music_filename[:-4],
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed Text2MusicWithTitle, Input Text: {text}, Output Music: {music_filename}.")
        return music_filename

class ReArrangement(object):
    def __init__(self, device):
        print("Initializing Text2MusicWithMelody")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Generate a new music arrangement with text indicating new style and previous music.",
        description="useful if you want to style transfer or rearrange music with a user input text describing the target style and the previous music."
                    "Please use Text2MusicWithDrum instead if the condition is a single drum track."
                    "You shall not use it when no previous music file in the history."
                    "like: remix the given melody with text description, or doing style transfer as text described from previous music."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the text description."
    )

    def inference(self, inputs):
        music_filename, text = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        attribute_table.descriptions = text
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

class Text2MusicWithDrum(object):
    def __init__(self, device):
        print("Initializing Text2MusicWithDrum")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Generate music from user input text based on the drum audio file provided.",
        description="useful if you want to generate music from a user input text and a previous given drum audio file."
                    "Do not use it when no previous music file (generated of uploaded) in the history."
                    "like: generate a pop song based on the provided drum pattern above."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the text description."
    )

    def inference(self, inputs):
        music_filename, text = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        text = description_to_attributes(text)
        print(f"Generating music from text with drum condition, Input text: {text}, Drum: {music_filename}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="withdrum")
        drum, sr = torchaudio.load(music_filename)
        self.model.set_generation_params(duration=35)
        wav = self.model.generate_continuation(prompt=drum[None].expand(GENERATION_CANDIDATE, -1, -1), prompt_sample_rate=sr,
                                               descriptions=[text] * GENERATION_CANDIDATE, progress=False)
        self.model.set_generation_params(duration=DURATION)
        # cut drum prompt
        wav = wav[..., int(drum.shape[-1] / sr * self.model.sample_rate):]
        splitted_audios = split_audio_tensor_by_downbeats(wav.cpu(), self.model.sample_rate, True)
        # select the best one by CLAP scores
        print(f"CLAP post filter for {len(splitted_audios)} candidates.")
        best_wav, _ = CLAP_post_filter(CLAP_model, text, splitted_audios.cuda(), self.model.sample_rate)
        print(f"\nProcessed Text2MusicWithDrum, Output Music: {updated_music_filename}.")
        return updated_music_filename


class AddNewTrack(object):
    def __init__(self, device):
        print("Initializing AddNewTrack")
        self.device = device
        self.model = musicgen_model

    @prompts(
        name="Add a new track to the given music loop",
        description="useful if you want to add a new track (usually add a new instrument) to the given music."
                    "like: add a saxophone to the given music, or add piano arrangement to the given music."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the text description."
    )

    def inference(self, inputs):
        music_filename, text = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        attribute_table.descriptions = merge_description(attribute_table.descriptions, text)
        text = addtrack_demand_to_description(text)
        print(f"Adding a new track, Input text: {text}, Previous track: {music_filename}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="addtrack")
        p_track, sr = torchaudio.load(music_filename)
        self.model.set_generation_params(duration=35)
        wav = self.model.generate_continuation(prompt=p_track[None].expand(GENERATION_CANDIDATE, -1, -1), prompt_sample_rate=sr,
                                               descriptions=[text] * GENERATION_CANDIDATE, progress=False)
        self.model.set_generation_params(duration=DURATION)
        # cut drum prompt
        wav = wav[..., int(p_track.shape[-1] / sr * self.model.sample_rate):]
        splitted_audios = split_audio_tensor_by_downbeats(wav.cpu(), self.model.sample_rate, True)
        # select the best one by CLAP scores
        print(f"CLAP post filter for {len(splitted_audios)} candidates.")
        best_wav, _ = CLAP_post_filter(CLAP_model, attribute_table.descriptions, splitted_audios.cuda(), self.model.sample_rate)
        best_wav = torch.from_numpy(librosa.effects.trim(best_wav.cpu().numpy())[0])
        audio_write(updated_music_filename[:-4],
                    best_wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed AddNewTrack, Output Music: {updated_music_filename}.")
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
            instrument_mode = instrument_mode_file = instrument
        elif mode == "remove":
            instrument_mode = f"no{instrument}"
            instrument_mode_file = f"no_{instrument}"
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
            os.path.join("separated", "htdemucs_6s", music_filename[:-4].split("/")[-1],f"{instrument_mode_file}.wav"),
            updated_music_filename
        )
        # delete the folder
        # os.system(f"rm -rf {os.path.join('separated', 'htdemucs_6s')}")

        print(f"Processed Source Separation, Input Music: {music_filename}, Output Instrument: {instrument_mode_file}.")
        return updated_music_filename


# class SimpleTracksMixing(object):
#     def __init__(self, device):
#         print("Initializing SimpleTracksMixing")
#         self.device = device
#
#     @prompts(
#         name="Simply mixing two tracks from two music files.",
#         description="useful if you want to mix two tracks from two music files."
#                     "Like: mix the vocals track from a music file with the drums track from another music file."
#                     "The input to this tool should be a comma separated string of two, "
#                     "representing the first music_filename_1 and the second music_filename_2."
#     )
#
#     def inference(self, inputs):
#         music_filename_1, music_filename_2 = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
#         print(f"Mixing two tracks from two music files, Input Music 1: {music_filename_1}, Input Music 2: {music_filename_2}.")
#         updated_music_filename = get_new_audio_name(music_filename_1, func_name="mixing")
#         # load
#         wav_1, sr_1 = torchaudio.load(music_filename_1)
#         wav_2, sr_2 = torchaudio.load(music_filename_2)
#         # resample
#         if sr_1 != sr_2:
#             wav_2 = torchaudio.transforms.Resample(sr_2, sr_1)(wav_2)
#         # pad or cut
#         if wav_1.shape[-1] > wav_2.shape[-1]:
#             wav_2 = torch.cat([wav_2, torch.zeros_like(wav_1[:, wav_2.shape[-1]:])], dim=-1)
#         elif wav_1.shape[-1] < wav_2.shape[-1]:
#             wav_2 = wav_2[:, :wav_1.shape[-1]]
#         # mix
#         assert wav_1.shape == wav_2.shape  # channel, length
#         wav = torch.add(wav_1, wav_2 * 0.7)
#         # write
#         audio_write(updated_music_filename[:-4],
#                     wav.cpu(), sr_1, strategy="loudness", loudness_compressor=True)
#         print(f"\nProcessed TracksMixing, Output Music: {updated_music_filename}.")
#         return updated_music_filename


class MusicCaptioning(object):
    def __init__(self):
        print("Initializing MusicCaptioning")

    @prompts(
        name="Describe the current music.",
        description="useful if you want to describe a music."
                    "Like: describe the current music, or what is the current music sounds like."
                    "The input to this tool should be the music_filename. "
    )

    def inference(self, inputs):
        music_filename = inputs.strip()
        print(f"Captioning the current music, Input Music: {music_filename}.")
        captions = captioning(music_filename)
        captions_text = captions[0]["text"]
        print(f"\nProcessed MusicCaptioning, Output Captions: {captions_text}.")
        return captions_text


# class Text2MusicwithChord(object):
#     template_model = True
#     def __init__(self, Text2Music):
#         print("Initializing Text2MusicwithChord")
#         self.Text2Music = Text2Music
#
#     @prompts(
#         name="Generate music from user input text and chord description",
#         description="useful only if you want to generate music from a user input text and explicitly mention a chord description."
#                     "Like: generate a pop love song with piano and a chord progression of C - F - G - C, or generate a sad music with a jazz chord progression."
#                     "This tool will automatically extract chord information and generate music."
#                     "The input to this tool should be the user input text. "
#     )
#
#     def inference(self, inputs):
#         music_filename = os.path.join("music", f"{str(uuid.uuid4())[:8]}.wav")
#
#         chords_list = chord_generation(inputs)
#         preprocessed_input = description_to_attributes(inputs)
#
#         for i, chord in enumerate(chords_list):
#             text = f"{preprocessed_input} key: {chord}."
#             self.Text2Music.model.set_generation_params(duration=(i + 1) * (DURATION / len(chords_list)))
#             if i == 0:
#                 wav = self.Text2Music.model.generate([text], progress=False)
#             else:
#                 wav = self.Text2Music.model.generate_continuation(wav,
#                                                                   self.Text2Music.model.sample_rate,
#                                                                   [text],
#                                                                   progress=False)
#                 if i == len(chords_list) - 1:
#                     wav = wav[0]  # batch size is 1
#                     audio_write(music_filename[:-4],
#                                 wav.cpu(), self.Text2Music.model.sample_rate, strategy="loudness", loudness_compressor=True)
#         self.Text2Music.model.set_generation_params(duration=DURATION)
#         print(f"\nProcessed Text2Music, Input Text: {preprocessed_input}, Output Music: {music_filename}.")
#         return music_filename

class PitchShifting(object):
    def __init__(self, device):
        print("Initializing PitchShifting")
        self.device = device

    @prompts(
        name="Shift the pitch of the given music.",
        description="useful if you want to shift the pitch of a music."
                    "Like: shift the pitch of this music by 3 semitones."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the pitch shift value."
    )

    def inference(self, inputs):
        music_filename, pitch_shift_value = inputs.split(",")[0].strip(), int(inputs.split(",")[1].strip())
        print(f"Shifting the pitch of the given music, Input Music: {music_filename}, Pitch Shift Value: {pitch_shift_value}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="pitchshifting")
        # load
        wav, sr = torchaudio.load(music_filename)
        # shift
        wav = torchaudio.functional.pitch_shift(wav, sr, pitch_shift_value)
        # write
        audio_write(updated_music_filename[:-4],
                    wav.cpu(), sr, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed PitchShifting, Output Music: {updated_music_filename}.")
        return updated_music_filename

class TimeStretching(object):
    def __init__(self, device):
        print("Initializing TimeStretching")
        self.device = device

    @prompts(
        name="Stretch the time of the given music.",
        description="useful if you want to stretch the time of a music."
                    "Like: stretch the time of this music by 1.5."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the time stretch value."
    )

    def inference(self, inputs):
        music_filename, time_stretch_value = inputs.split(",")[0].strip(), float(inputs.split(",")[1].strip())
        print(f"Stretching the time of the given music, Input Music: {music_filename}, Time Stretch Value: {time_stretch_value}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="timestretching")
        # load
        wav, sr = torchaudio.load(music_filename)
        # stretch
        wav = torchaudio.functional.speed(wav, sr, time_stretch_value)[0]
        # write
        audio_write(updated_music_filename[:-4],
                    wav.cpu(), sr, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed TimeStretching, Output Music: {updated_music_filename}.")
        return updated_music_filename

class MusicInpainting(object):
    def __init__(self, device):
        print("Initializing MusicInpainting")
        self.device = device
        self.interface = interface

    @prompts(
        name="Inpaint a specific time region of the given music.",
        description="useful if you want to inpaint or regenerate a specific region (must with explicit time start and ending) of music."
                    "like: re-generate the 3s-5s part of this music."
                    "The input to this tool should be a comma separated string of three, "
                    "representing the music_filename, the start time (in second), and the end time (in second)."
    )

    def inference(self, inputs):
        music_filename, start_time, end_time = inputs.split(",")[0].strip(), inputs.split(",")[1].strip(), inputs.split(",")[2].strip()
        print(f"Inpainting a specific time region of the given music, Input Music: {music_filename}, Start Time: {start_time}, End Time: {end_time}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="inpainting")
        p_track, sr = torchaudio.load(music_filename)
        audio_length_in_second = p_track.shape[-1] / sr
        if float(end_time) > audio_length_in_second:
            print(f"Invalid end time, please check the input.")
            end_time = audio_length_in_second
        start_time, end_time = int(start_time), int(audio_length_in_second - float(end_time))
        vamp(input_audio_path=music_filename,
             output_audio_path=updated_music_filename,
             interface=self.interface,
             prefix_s=start_time,
             suffix_s=end_time)
        print(f"\nProcessed MusicInpainting, Output Music: {updated_music_filename}.")
        return updated_music_filename

class Variation(object):
    def __init__(self, device):
        print("Initializing Variation")
        self.device = device
        self.interface = interface

    @prompts(
        name="Generate a variation of given music.",
        description="useful if you want to generate a variation of music, or re-generate the entire music track."
                    "like: re-generate this music, or, generate a variant."
                    "The input to this tool should be a single string, "
                    "representing the music_filename."
    )

    def inference(self, inputs):
        music_filename = inputs
        print(f"Generate a variation of given music, Input Music: {music_filename}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="variation")
        p_track, sr = torchaudio.load(music_filename)
        vamp(input_audio_path=music_filename,
             output_audio_path=updated_music_filename,
             interface=self.interface,)
        print(f"\nProcessed Variation, Output Music: {updated_music_filename}.")
        return updated_music_filename

class SingleSoundEffect(object):
    def __init__(self, device):
        print("Initializing SingleSoundEffect")
        self.device = device
        self.interface = interface

    @prompts(
        name="Add a single sound effect to the given music.",
        description="useful if you want to add a single sound effect, like reverb, high pass filter or chorus to the given music."
                    "like: add a reverb of recording studio to this music."
                    "The input to this tool should be a comma separated string of two, "
                    "representing the music_filename and the original user message."
    )

    def inference(self, inputs):
        music_filename, user_message = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
        print(f"Add a single sound effect to the given music, Input Music: {music_filename}, Sound Effect Name: {user_message}.")
        updated_music_filename = get_new_audio_name(music_filename, func_name="single_sound_effect")
        sound_effect = add_single_sound_effect(user_message)
        my_pedalboard = pedalboard.Pedalboard()
        my_pedalboard.append(eval(sound_effect))
        input_audio, sr = torchaudio.load(music_filename)
        output_audio = my_pedalboard(input_audio.numpy(), sample_rate=sr)
        audio_write(updated_music_filename[:-4],
                    output_audio, sr, strategy="loudness", loudness_compressor=True)
        print(f"\nProcessed SingleSoundEffect, Output Music: {updated_music_filename}.")
        return updated_music_filename


# class TimbreTransfer(object):
#     def __init__(self, device):
#             print("Initializing TimbreTransfer")
#             self.device = device
#             self.interface = interface
#
#         @prompts(
#             name="Transfer the timbre of the given music to another music.",
#             description="useful if you want to transfer the timbre of the given music to another music."
#                         "like: transfer the timbre of this music to another music."
#                         "The input to this tool should be a comma separated string of two, "
#                         "representing the music_filename and the original user message."
#         )
#
#     def inference(self, inputs):
#         music_filename, user_message = inputs.split(",")[0].strip(), inputs.split(",")[1].strip()
#         print(f"Transfer the timbre of the given music to another music, Input Music: {music_filename}, Target Music: {user_message}.")
#         updated_music_filename = get_new_audio_name(music_filename, func_name="timbre_transfer")
#         target_music_filename = get_new_audio_name(user_message, func_name="timbre_transfer")
#         # load
#         wav, sr = torchaudio.load(music_filename)
#         target_wav, target_sr = torchaudio.load(user_message)
#         # stretch
#         wav = torchaudio.functional.time_stretch(wav, sr, target_sr/sr)[0]
#         # write
#         audio_write(updated_music_filename[:-4],
#                     wav.cpu(), sr, strategy="loudness", loudness_compressor=True)
#         audio_write(target_music_filename[:-4],
#                     target_wav.cpu(), target_sr, strategy="loudness", loudness_compressor=True)
#         print(f"\nProcessed TimbreTransfer, Output Music: {updated_music_filename}.")
#         return updated_music_filename
