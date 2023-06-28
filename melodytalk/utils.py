import random
import uuid
import torch
import numpy as np
import os
import openai
import typing as tp

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

def get_new_audio_name(org_audio_name: str, func_name: str ="update") -> str:
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

    openai_prompt = f"""Please catch the bpm and key attributes from the original description text. If the description text does not mention it, do not add it. Here are two examples:

    Q: Generate a love pop song in C major of 120 bpm.
    A: Generate a love pop song. bpm: 120. key: Cmaj. 
    
    Q: Generate a love pop song in a minor.
    A:  Generate a love pop song. key: Amin. 
    
    Q: {description}.
    A: 
    """

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=openai_prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["\n"]
    )

    return response.choices[0].text


def chord_generation(description: str, chord_num: int = 4) -> tp.List:
    """ This function is a trick to generate chord sequence from the description.

    :param description:
    :param chord_num:
    :return:
    """

    openai_prompt = f"""Please generate a chord sequence consists of according to the text description. Example:

    Q: Generate a chord sequence for sad pop song. 4 chords.
    A: Dm - Bb - F - C
    
    Q: {description}. {chord_num} chords.
    A:
    """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=openai_prompt,
        temperature=1,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    chord_list = [i.strip() for i in response.choices[0].text.split('-')]

    return chord_list
