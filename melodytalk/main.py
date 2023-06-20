import gradio as gr
import argparse
import uuid
import inspect
import tempfile
import numpy as np
import torch
import os
import re

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI


MELODYTALK_PREFIX = """MelodyTalk is designed to be able to assist with a wide range of text and music related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. MelodyTalk is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

MelodyTalk is able to process and understand large amounts of text and music. As a language model, MelodyTalk can not directly read audios, but it has a list of tools to finish different music and audio tasks. Each audio will have a file name formed as "audio/xxx.png", and MelodyTalk can invoke different tools to indirectly understand audios. When talking about audios, MelodyTalk is very strict to the file name and will never fabricate nonexistent files. 

MelodyTalk is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the audio content and audio file name. It will remember to provide the file name from the last tool observation, if a new audio is generated.

Human may provide new audios to MelodyTalk with a description. The description helps MelodyTalk to understand this audio, but MelodyTalk should use tools to finish following tasks, rather than directly imagine from the description.

Overall, MelodyTalk is a powerful audio dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

MelodyTalk has access to the following tools:"""

MELODYTALK_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

MELODYTALK_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the audio file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since MelodyTalk is a text language model, MelodyTalk must use tools to observe audios rather than imagination.
The thoughts and observations are only visible for MelodyTalk, MelodyTalk should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

MELODYTALK_PREFIX_CN = """MelodyTalk被设计成能够协助完成各种文本和音乐相关的任务，从回答简单的问题到提供深入的解释和对各种主题的讨论。MelodyTalk能够根据其收到的输入生成类似人类的文本，使其能够参与自然的对话，并提供与当前主题相关的连贯的回应。

MelodyTalk能够处理和理解大量的文本和音乐。作为一个语言模型，MelodyTalk不能直接阅读音频，但它有一系列工具来完成不同的音乐和音频任务。每个音频都会有一个文件名，形成 "audio/xxx.png"，MelodyTalk可以调用不同的工具来间接理解音频。当谈及音频时，MelodyTalk对文件名的要求非常严格，绝不会编造不存在的文件。

MelodyTalk能够按顺序使用工具，并忠于工具观察输出，而不是伪造音频内容和音频文件名。如果有新的音频产生，它将记得提供上一个工具观察的文件名。

人类可以向MelodyTalk提供带有描述的新音频。描述可以帮助MelodyTalk理解这个音频，但是MelodyTalk应该使用工具来完成以下任务，而不是直接从描述中想象。

总的来说，MelodyTalk是一个强大的音频对话助手工具，可以帮助完成各种任务，并提供关于各种主题的宝贵见解和信息。

工具列表:
------

MelodyTalk 可以使用这些工具:"""

MELODYTALK_FORMAT_INSTRUCTIONS_CN = """用户使用中文和你进行聊天，但是工具的参数应当使用英文。如果要调用工具，你必须遵循如下格式:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

当你不再需要继续调用工具，而是对观察结果进行总结回复时，你必须使用如下格式：


```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

MELODYTALK_SUFFIX_CN = """你对文件名的正确性非常严格，而且永远不会伪造不存在的文件。

开始!

因为MelodyTalk是一个文本语言模型，必须使用工具去观察音频而不是依靠想象。
推理想法和观察结果只对MelodyTalk可见，需要记得在最终回复时把重要的信息重复给用户，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""

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

def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
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
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)

class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if
                                           k != 'self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang):
        self.memory.clear()  # clear previous history
        if lang == 'English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = MELODYTALK_PREFIX, MELODYTALK_FORMAT_INSTRUCTIONS, MELODYTALK_SUFFIX
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = MELODYTALK_PREFIX_CN, MELODYTALK_FORMAT_INSTRUCTIONS_CN, MELODYTALK_SUFFIX_CN
            place = "输入文字并回车，或者上传图片"
            label_clear = "清除"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible=True), gr.update(visible=False), gr.update(placeholder=place), gr.update(
            value=label_clear)

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_image(self, image, state, txt, lang):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = False#Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        if lang == 'Chinese':
            Human_prompt = f'\nHuman: 提供一张名为 {image_filename}的图片。它的描述是: {description}。 这些信息帮助你理解这个图像，但是你应该使用工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。  "
        else:
            Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
            AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state, f'{txt} {image_filename} '

if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="Text2Music_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        lang = gr.Radio(choices = ['Chinese','English'], value=None, label='Language')
        chatbot = gr.Chatbot(elem_id="chatbot", label="MelodyTalk")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an audio").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️",file_types=["audio"])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0", server_port=7860)