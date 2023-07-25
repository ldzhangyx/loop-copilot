import random
import uuid
import torch
import numpy as np
import os
import openai
import typing as tp
import madmom
import resampy
import torchaudio
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY")


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def get_new_audio_name(org_audio_name: str, func_name: str = "update") -> str:
    head_tail = os.path.split(org_audio_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.wav'
    return os.path.join(head, new_file_name)


def description_to_attributes(description: str) -> str:
    """ This function is a trick to concate key, bpm, (genre, mood, instrument) information to the description.

    :param description:
    :return:
    """

    openai_prompt = f"""Please: 1. split and paste the bpm and key attributes from the original description text; 
    copy and paste the instrument attribute but leave the original text; delete the chord description if necessary. keep the rest unchanged. 
    If the description text does not mention it, do not add it. Here are some examples:

    Q: Generate a love pop song with piano and violin in C major of 120 bpm.
    A: a love pop song with piano and violin. bpm: 120. key: C major. instrument: piano, violin.
    
    Q: love pop song in A minor, creating a romantic atmosphere.
    A: love pop song, creating a romantic atmosphere. key: A minor. 
    
    Q: Generate a pop song with chord progression of C - G - Am - F, with piano and guitar.
    A: a pop song with piano and guitar. instrument: piano, guitar.
    
    Q: {description}.
    A: """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response.choices[0].text


def addtrack_demand_to_description(description: str) -> str:
    openai_prompt = f"""Please rewrite the sentence in the following format.

    Q: Please add a saxophone track to this music.
    A: music loop with saxophone track.

    Q: add some woodwind arrangement.
    A: music loop with woodwind arrangement.

    Q: {description}.
    A: """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    return response.choices[0].text


def chord_generation(description: str) -> tp.List:
    """ This function is a trick to generate chord sequence from the description.

    :param description:
    :param chord_num:
    :return:
    """

    openai_prompt = f"""Please analyse and generate a chord sequence (4 chords by default) according to the text description. Example:

    # random generate chord sequence
    Q: Generate a song which has a jazz chord progression, with piano and guitar. 
    A: D minor - Bb major - F major - C major
    
    # formalize chord description
    Q: Generate a pop song which has a chord progression of I - IV - V - I, with piano and guitar. 
    A: C major - F major - G major - C major
    
    Q: {description}
    A: """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        temperature=1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    chord_list = [i.strip() for i in response.choices[0].text.split(' - ')]

    return chord_list


def beat_tracking_with_clip(audio_path: str,
                            output_path: str = None,
                            offset: int = 0,
                            bar: int = 4,
                            beat_per_bar: int = 4, ):
    proc = madmom.features.beats.DBNDownBeatTrackingProcessor(beats_per_bar=beat_per_bar, fps=100)
    beats = proc(audio_path)
    # we cut the audio to only bar * beat_per_bar beats, and shift the first beat to offset
    first_beat_time = beats[0][0]
    last_beat_time = beats[bar * beat_per_bar][0]  # the beginning of the next bar
    begin_time_with_offset = first_beat_time + offset
    end_time = last_beat_time + offset
    # cut the audio clip
    audio = AudioSegment.from_wav(audio_path)
    audio_clip = audio[begin_time_with_offset * 1000: end_time * 1000]
    if output_path is None:
        output_path = audio_path
    audio_clip.export(output_path, format="wav")


def split_track_with_beat(input_track):
    pass


@torch.no_grad()
def CLAP_post_filter(clap_model,
                     text_description: str,
                     audio_candidates: torch.Tensor,
                     audio_sr: int) \
        -> torch.Tensor and int:
    """ This function is a post filter for CLAP model. It takes the text description and audio candidates as input,
    and returns the most similar audio and its similarity score.

    args:
        clap_model: CLAP model
        text_description: the text description of the audio
        audio_candidates: the audio candidates
        audio_sr: the sample rate of the audio candidates

    return:
        audio_embedding: the embedding of the audio candidates
        similarity: the similarity score
    """
    # if audio will in shape [N, C, L], then make C axis average. Usually C = 1.
    audio_candidates = torch.mean(audio_candidates, dim=1)
    # resample the audio_candidates to 48k which supports CLAP model
    audio_candidates = resampy.resample(audio_candidates.cpu().numpy(), audio_sr, 48000, axis=-1)
    audio_candidates = torch.from_numpy(audio_candidates)
    # calculate the audio embedding
    audio_embedding = clap_model.get_audio_embedding_from_data(x=audio_candidates, use_tensor=True)  # (N, D)
    # calculate the text embedding
    text_embedding = clap_model.get_text_embedding([text_description] * audio_embedding.size(dim=0), use_tensor=True)  # (N, D)
    # calculate the similarity by dot product
    similarity = torch.sum(audio_embedding * text_embedding, dim=-1) # (N,)
    # get the index of the most similar audio
    index = torch.argmax(similarity)
    best = audio_candidates[index].view(1, -1)
    best = resampy.resample(best.cpu().numpy(), 48000, audio_sr, axis=-1)
    # return
    return best, similarity[index]


def split_audio_tensor_by_downbeats(input_audio_batch: torch.Tensor, sr: int = 36000,
                                    return_stack: bool = True) -> torch.Tensor:
    segments = []
    bars = 4
    act = madmom.features.downbeats.RNNDownBeatProcessor()
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    for i, input_audio in enumerate(input_audio_batch):
        # save to temp file
        temp_file_name = f'cache/temp_{i}.wav'
        temp_file_path = os.path.join(os.getcwd(), temp_file_name)
        torchaudio.save(temp_file_path, input_audio, sr)

        # estimation
        all_beats = proc(act(temp_file_path))
        # split
        upbeats = [i[0] for i in all_beats if i[1] == 1]
        upbeats_index = [int(i * sr) for i in upbeats][::bars]
        for j in range(len(upbeats_index) - 1):
            segments.append(input_audio[..., upbeats_index[j]: upbeats_index[j + 1]])

    if return_stack:
        # pad
        max_len = max([i.shape[-1] for i in segments])
        segments = [torch.nn.functional.pad(i, (0, max_len - i.shape[-1])) for i in segments]
        # stack
        segments = torch.stack(segments)

    return segments



