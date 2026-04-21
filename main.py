import asyncio
import sys
import threading
from xmlrpc import client

import torch
import numpy as np

from PIL import ImageGrab
import time
from collections import deque

import base64
from io import BytesIO
from httpcore import URL
import pytesseract
import requests
import qasync
from openai import AsyncOpenAI
import json
import yaml
import queue

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QPoint, QTimer 
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QObject, pyqtSignal

import pyaudio
from faster_whisper import WhisperModel


def pil_image_to_base64(pil_image):
    buffered = BytesIO() # 制造一个存在于内存里的“虚拟文件”
    pil_image.save(buffered, format="PNG") # 把内存里的图片对象，存进这个虚拟文件里（指定格式为 PNG）

    # 提取虚拟文件里的二进制数据，打包成 base64 文本
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_to_base64_by_path(image_path):
    # 以二进制读取模式("rb")打开图片，并转换为 base64 文本
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def pack_msg(role, type, content, tool_call=None):
    if type == "text":
        return {"role": role, "content": content}
    elif type == "image_url":
        return {"role": role, "content": [{"type": "image_url", "image_url": {"url": content}}]}
    elif type == "tool":
        return {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": content
            }

class Vision: #AI的视觉模块
    def __init__(self, history_length=5):
        self.history = deque(maxlen=history_length)

    def update(self, screenshot):
        self.history.append(screenshot)
        
    def sudden_view(self):
        self.update(ImageGrab.grab())
        return self.history[-1]

    async def look_at_screen(self):
        try:
            screenshot = self.sudden_view()
            screen_base64 = pil_image_to_base64(screenshot)
            img_msg = pack_msg("user", "image_url", f"data:image/png;base64,{screen_base64}")

            return img_msg
            
        except Exception as e:
            print(f"识别失败：{e}")
            return "糟糕，本堂主的眼睛出了点问题，看不清屏幕了。"

class Listen(QObject): 
    text_signal = pyqtSignal(str) #只是信号通道，不是消息缓存。

    def __init__(self, history_length=5):
        super().__init__()
        self.listen_history = deque(maxlen=history_length)

        self.audio = pyaudio.PyAudio()

        # 音频配置参数 (VAD 要求的标准格式)
        self.SAMPLE_RATE = 16000 # 采样率：16kHz
        self.CHUNK = 512         # 每次读取的音频块大小

        # 开启麦克风数据流
        self.stream = self.audio.open(
            format=pyaudio.paInt16, # 16位深度
            channels=1,             # 单声道
            rate=self.SAMPLE_RATE,  # 16000 Hz
            input=True,             # 这是一个输入流（麦克风）
            frames_per_buffer=self.CHUNK # 每次读取 512 帧
        )

        # 从 PyTorch Hub 自动加载 Silero VAD
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0] # 获取处理工具

        #加载whisper
        self.whisper_model = WhisperModel("small", device="cuda", compute_type="float16")

        self.start_threading()

    def start_threading(self):
        listen_thread = threading.Thread(target=self.during_listening, daemon=True) #daemon=True意思是：这个线程是个守护线程，主线程结束了它也会跟着结束，不会阻碍程序退出。
        listen_thread.start()

    def get_voice_text(self, text):
        self.text_signal.emit(text) #广播机制，并不需要传给谁，只要发出这个信号了，任何监听这个信号的对象都能收到并处理这个文本消息了
    
    def during_listening(self):
        silence_timeout = 0
        MAX_SILENCE = 20
        while True:
            #首先需要完成一个字节流-》numpy-》tensor的转换
            voice_buffer = []
            raw_bytes = self.stream.read(self.CHUNK)
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16).copy()
            tensor_chunk = torch.from_numpy(audio_data).float() / 32768.0
            #模型打分
            score = self.model(tensor_chunk, self.SAMPLE_RATE).item()
            if score >= 0.5:
                print("检测到声音了，开始录音...")
                voice_buffer.append(raw_bytes)
                while silence_timeout < MAX_SILENCE:
                    raw_bytes = self.stream.read(self.CHUNK)
                    audio_data = np.frombuffer(raw_bytes, dtype=np.int16).copy()
                    tensor_chunk = torch.from_numpy(audio_data).float() / 32768.0

                    #模型打分
                    score = self.model(tensor_chunk, self.SAMPLE_RATE).item()

                    if score < 0.5:
                        silence_timeout += 1

                    else:
                        silence_timeout = 0

                    
                    voice_buffer.append(raw_bytes)
                
                print("录音结束，正在转写...")
                complete_audio_bytes = b"".join(voice_buffer)
                complele_audio_np = np.frombuffer(complete_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # 把 complete_audio_bytes 交给 Whisper 转写
                segments, info = self.whisper_model.transcribe(complele_audio_np, beam_size=5, language="zh")
                transed_text = "".join([segment.text for segment in segments])
                self.get_voice_text(transed_text)
                silence_timeout = 0

class Brain:
    def __init__(self): 
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.context = deque(maxlen=5)
        self.cur_model = "gemini-2.5-pro"
        self.client = AsyncOpenAI(api_key=config["API_KEY"], base_url=config["BASE_URL"])
        self.tools = [
            {
                "type": "function",
                "function":{
                    "name": "look_at_screen",
                    "description": "look at the current screen",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    async def get_llm_response(self, message, model=None):
        '''
        message和response_msg都会直接存入context。
        '''
        self.context.append({"role": "user", "content": message})
        if model is None:
            model = self.cur_model

        response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant.中文回复"},
                    *self.context,
                ],
                # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
                extra_body={"reasoning_split": True},
                tools=self.tools,
                tool_choice="auto"
            )
        response_msg = {"role": response.choices[0].message.role, "content": response.choices[0].message.content}
        self.context.append(response_msg)
        # resp_message = response.choices[0].message
        return response

    async def get_response_with_context(self, context, model=None):
        if model is None:
            model = self.cur_model
        response = await self.client.chat.completions.create(
                model=model,
                messages=context,
                # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
                extra_body={"reasoning_split": True},
                tools=self.tools,
                tool_choice="auto"
            )
        # resp_message = response.choices[0].message
        return response

class Hutao(QWidget):
    def __init__(self):
        super().__init__()
        '''
        中枢神经系统：负责宏观调控
        '''

        #UI部分
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_trick)
        self.timer.start(10000)  # 10秒触发一次

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground) #设置透明背景

        self.label = QLabel(self)
        pixmap = QPixmap("hutao.jpg")
        pixmap = pixmap.scaledToWidth(150, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height()) #让窗口大小和图片匹配

        self.drag_poision = QPoint()

        #器官部分
        self.brain = Brain()
        self.vision = Vision()
        self.listen = Listen()

        # 连接听觉信号到处理函数
        self.listen.text_signal.connect(self.on_heard_text) #发射器.信号.connect(接收器)

        #自锁
        self.is_busy = False

        #消息队列-》使用生产者-消费者结构实现
        self.message_queue = asyncio.Queue()
        asyncio.get_event_loop().create_task(self.on_received_message_consumer()) #创建一个后台任务，专门负责消费消息队列里的消息
        
    async def do_response(self, message):
        response = await self.brain.get_llm_response(message)
        resp_message = response.choices[0].message
        while resp_message.tool_calls:
            tool_call = resp_message.tool_calls[0]
            print(f"接收到军师指令，准备运行: {tool_call.function.name}")
            tool_result = await self.tool_executer(tool_call)

            # 按照标准格式，把执行结果打包
            tool_msg = pack_msg("tool", "tool", tool_result, tool_call)

            # 第二次通信：带着结果回去要最终回复
            response = await self.brain.get_llm_response(tool_msg)
            resp_message = response.choices[0].message
        
        print(resp_message.content)

    async def tool_executer(self, tool_call):
        # target_method = getattr(self, tool_call.function.name)
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
        try:
            # 尝试解析
            args_dict = json.loads(args_str) if args_str else {}
            # 如果解析出来是 None (比如遇到了 "null")，或者不是字典，强制兜底为空字典
            if not isinstance(args_dict, dict):
                args_dict = {}
        except Exception:
            # 万一大模型抽风发来一段根本无法解析的乱码，也用空字典兜底
            args_dict = {}

        if func_name == "look_at_screen":
            img_msg = await self.vision.look_at_screen()
            self.brain.context.append(img_msg)
            return "已查看屏幕并将图片信息加入上下文了哦"

    async def should_reply(self, message):
        try:
            judge_msg = pack_msg("system", "text", "你是一个聪明的助手，负责判断用户的消息是否需要回复。如果需要回复，回复true；如果不需要回复，回复false。")
            user_msg = pack_msg("user", "text", f"用户的消息是：{message}")
            judge_context = [judge_msg, user_msg]

            response = await self.brain.get_response_with_context(judge_context)
            
            reply_decision = response.choices[0].message.content.strip().lower()
            return reply_decision == "true"
        except Exception as e:
            print(f"判断是否回复时出错了：{e}")
            return False

    @qasync.asyncSlot(str) #asyncSlot意思是：这个函数虽然是协程函数，但我要把它当成 Qt 的槽函数来用。普通的 PyQt 只认识同步槽函数，不会自动 await 你的协程。qasync 的这个装饰器会帮你把协程正确地挂到事件循环里执行。(str)意思是：这个槽函数期望接收一个 str 类型的信号参数。
    async def on_heard_text(self, text):
        print(f"接收到听觉消息：{text}")
        if not self.is_busy:
            await self.message_queue.put(text) #把消息放到队列里，等着消费者去处理
        else:
            print("当前忙碌，暂时无法处理新的消息。")

    # async def on_received_message_producer(self, message):
    #     self.messsage_queue.put(message)

    async def on_received_message_consumer(self):
        while True:
            message = await self.message_queue.get() #当队列为空时就会永远停留在这一行
            self.is_busy = True
            try:
                print(f"正在处理消息：{message}")
                if await self.should_reply(message):
                    print("判断需要回复，正在处理消息...")
                    await self.do_response(message)
                    # print(response.choices[0].message.content)
            except Exception as e:
                print(f"处理消息时出错了：{e}")
            finally:
                self.is_busy = False
                # self.message_queue.task_done() #和join成对出现，join表示开始，task_done表示结束。虽然现在这个队列没有调用join，但为了保持结构的完整性，还是调用一下task_done吧。

    def on_timer_trick(self):
        # resp = self.get_response("请你调用工具看看我的屏幕")
        # print(resp.choices[0].message.content)
        self.vision.sudden_view()

    def mouseDoubleClickEvent(self, event): #鼠标双击时
        if event.button() == Qt.LeftButton:
            print("接收到鼠标双击事件")
            self.message_queue.put_nowait("用户用鼠标触碰了你")
            event.accept()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_poision = event.globalPos() - self.frameGeometry().topLeft() #以左上角为偏移点
            event.accept()

    def mouseMoveEvent(self, event): #鼠标拖动时
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_poision)
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app) #创建兼容PyQt的异步事件
    asyncio.set_event_loop(loop) #设置异步事件循环

    hutao = Hutao()
    hutao.show()
    
    while loop:
        loop.run_forever()