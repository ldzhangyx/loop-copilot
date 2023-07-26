import gradio as gr

# audio processing

# langchain interface
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI

# MusicGen

from modules import *

MELODYTALK_PREFIX = """MelodyTalk is designed to be able to assist with a wide range of text and music related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. MelodyTalk is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

MelodyTalk is able to process and understand large amounts of text and music. As a language model, MelodyTalk can not directly read music, but it has a list of tools to finish different music tasks. Each music will have a file name formed as "music/xxx.wav", and MelodyTalk can invoke different tools to indirectly understand music. When talking about music, MelodyTalk is very strict to the file name and will never fabricate nonexistent files. 

MelodyTalk is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the music content and music file name. It will remember to provide the file name from the last tool observation, if a new music is generated.

Human may provide new music to MelodyTalk with a description. The description helps MelodyTalk to understand this music, but MelodyTalk should use tools to finish following tasks, rather than directly imagine from the description.

Overall, MelodyTalk is a powerful music dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

MelodyTalk has access to the following tools:"""

MELODYTALK_FORMAT_INSTRUCTIONS = """To use a tool, you MUST use the following format:

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
You will remember to provide the music file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

Since MelodyTalk is a text language model, MelodyTalk must use tools to observe music rather than imagination. The thoughts and observations are only visible for MelodyTalk.

New input: {input}
Thought: Do I need to use a tool? {agent_scratchpad} You MUST strictly follow the format.
"""

# removed from melodytalk_suffix:
# MelodyTalk should remember to repeat important information in the final response for Human.

MELODYTALK_PREFIX_CN = """MelodyTalk被设计成能够协助完成各种文本和音乐相关的任务，从回答简单的问题到提供深入的解释和对各种主题的讨论。MelodyTalk能够根据其收到的输入生成类似人类的文本，使其能够参与自然的对话，并提供与当前主题相关的连贯的回应。

MelodyTalk能够处理和理解大量的文本和音乐。作为一个语言模型，MelodyTalk不能直接阅读音乐，但它有一系列工具来完成不同的音乐任务。每个音乐都会有一个文件名，形成 "music/xxx.wav"，MelodyTalk可以调用不同的工具来间接理解音乐。当谈及音乐时，MelodyTalk对文件名的要求非常严格，绝不会编造不存在的文件。

MelodyTalk能够按顺序使用工具，并忠于工具观察输出，而不是伪造音乐内容和音乐文件名。如果有新的音乐产生，它将记得提供上一个工具观察的文件名。

人类可以向MelodyTalk提供带有描述的新音乐。描述可以帮助MelodyTalk理解这个音乐，但是MelodyTalk应该使用工具来完成以下任务，而不是直接从描述中想象。

总的来说，MelodyTalk是一个强大的音乐对话助手工具，可以帮助完成各种任务，并提供关于各种主题的宝贵见解和信息。

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

因为MelodyTalk是一个文本语言模型，必须使用工具去观察音乐而不是依靠想象。
推理想法和观察结果只对MelodyTalk可见，你只能给用户返回中文句子。我们一步一步思考。在你使用工具时，工具的参数只能是英文。

聊天历史:
{chat_history}

新输入: {input}
Thought: Do I need to use a tool? {agent_scratchpad}
"""


class ConversationBot(object):
    def __init__(self):
        load_dict = {"Text2Music": "cuda:0",
                     "ExtractTrack": "cuda:0",
                     "Text2MusicWithMelody": "cuda:0",
                     "Text2MusicWithDrum": "cuda:0",
                     "AddNewTrack": "cuda:0"}
        template_dict = None  # { "Text2MusicwithChord": "cuda:0"} # "Accompaniment": "cuda:0",

        print(f"Initializing MelodyTalk, load_dict={load_dict}, template_dict={template_dict}")

        self.models = {}

        # global attribute table
        self.attribute_table = GlobalAttributes()

        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        # for class_name, device in template_dict.items():
        #     template_required_names = {k for k in inspect.signature(globals()[class_name].__init__).parameters.keys() if
        #                                k != 'self'}
        #     loaded_names = set([type(e).__name__ for e in self.models.values()])
        #     if template_required_names.issubset(loaded_names):
        #         self.models[class_name] = globals()[class_name](
        #             **{name: self.models[name] for name in template_required_names})

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
            place = "Enter text and press enter, or upload an audio"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = MELODYTALK_PREFIX_CN, MELODYTALK_FORMAT_INSTRUCTIONS_CN, MELODYTALK_SUFFIX_CN
            place = "输入文字并回车，或者上传音乐"
            label_clear = "清除"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible=True), gr.update(visible=False), gr.update(placeholder=place), gr.update(
            value=label_clear), gr.update(visible=True)

    def run_text(self, text, state):
        # LangChain has changed its implementation, so we are not able to cut the dialogue history anymore.
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        state = state + [(text, res['output'])]
        if len(res['intermediate_steps']) > 0:
            audio_filename = res['intermediate_steps'][-1][1]
            state = state + [(None, (audio_filename,))]
        # print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_audio(self, file, state, txt, lang):
        music_filename = os.path.join('music', str(uuid.uuid4())[0:8] + ".wav")
        print("Inputs:", file, state)
        if not isinstance(file, str):  # recording pass a path, while button pass a file object
            file = file.name
        audio_load, sr = torchaudio.load(file)
        audio_write(music_filename[:-4], audio_load, sr, strategy="loudness", loudness_compressor=True)
        # description = self.models['ImageCaptioning'].inference(image_filename)
        if lang == 'Chinese':
            Human_prompt = f'提供一个名为 {music_filename}的音乐。' \
                           f'这些信息帮助你理解这个音乐，但是你应该使用工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\".'
            AI_prompt = "收到。  "
        else:
            Human_prompt = f'Provide a music named {music_filename}. ' \
                           f'This information helps you to understand this music, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\".'
            AI_prompt = "Received.  "
        self.agent.memory.chat_memory.add_user_message(Human_prompt)
        self.agent.memory.chat_memory.add_ai_message(AI_prompt)
        state = state + [((music_filename,), AI_prompt)]
        # print(f"\nProcessed run_audio, Input music: {music_filename}\nCurrent state: {state}\n"
        #       f"Current Memory: {self.agent.memory.buffer}")
        return state, state

    def run_recording(self, file_path, state, txt, lang):

        return self.run_audio(file_path, state, txt, lang)

    def clear_input_audio(self):
        return gr.Audio.update(value=None)


if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    bot = ConversationBot()

    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:

        gr.Markdown(
            """This is a demo to our work *MelodyTalk*.
            """
        )

        lang = gr.Radio(choices=['Chinese', 'English'], value=None, label='Language')
        chatbot = gr.Chatbot(elem_id="chatbot", label="MelodyTalk")
        state = gr.State([])

        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an audio").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton("Upload", file_types=["audio"])

        with gr.Row(visible=False) as record_raws:
            with gr.Column(scale=0.7):
                rec_audio = gr.Audio(source='microphone', type='filepath', interactive=True, show_label=False)
            with gr.Column(scale=0.15, min_width=0):
                rec_clear = gr.Button("Re-recording")
            with gr.Column(scale=0.15, min_width=0):
                rec_submit = gr.Button("Submit")

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear, record_raws])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_audio, [btn, state, txt, lang], [chatbot, state])

        rec_submit.click(bot.run_recording, [rec_audio, state, txt, lang], [chatbot, state])
        rec_clear.click(bot.clear_input_audio, None, rec_audio)

        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
        clear.click(bot.clear_input_audio, None, rec_audio)
    demo.launch(server_name="0.0.0.0", server_port=7860,
                ssl_certfile="cert.pem", ssl_keyfile="key.pem", ssl_verify=False)
