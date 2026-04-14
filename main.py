import sys
from xmlrpc import client

from PIL import ImageGrab
import time
from collections import deque

import base64
from io import BytesIO
from httpcore import URL
import pytesseract
import requests
from openai import OpenAI
import json
from PetWindow import PetWindow
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt, QPoint, QTimer 
from PyQt5.QtGui import QPixmap

def pil_image_to_base64(pil_image):
    buffered = BytesIO() # 制造一个存在于内存里的“虚拟文件”
    pil_image.save(buffered, format="PNG") # 把内存里的图片对象，存进这个虚拟文件里（指定格式为 PNG）

    # 提取虚拟文件里的二进制数据，打包成 base64 文本
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def encode_image_to_base64_by_path(image_path):
    # 以二进制读取模式("rb")打开图片，并转换为 base64 文本
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class Vision: #AI的视觉模块
    def __init__(self, history_length=5):
        self.history = deque(maxlen=history_length)

    def update(self, screenshot):
        self.history.append(screenshot)
        
    def sudden_view(self):
        self.update(ImageGrab.grab())
        return self.history[-1]

class Hutao(QWidget):
    def __init__(self):
        super().__init__()

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

        self.frag_poision = QPoint()

        #llm调用部分
        URL = ""
        API_KEY= ""
        self.vision = Vision()
        self.context = deque(maxlen=5)
        self.cur_model = "gemini-2.5-pro"
        self.client = OpenAI(api_key=API_KEY, base_url=URL)
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
    
    def on_timer_trick(self):
        resp = self.get_response("请你调用工具看看我的屏幕")
        print(resp.choices[0].message.content)

    def get_response(self, message):
        self.context.append({"role": "user", "content": message})

        response = self.client.chat.completions.create(
                model=self.cur_model,
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
        resp_message = response.choices[0].message

        while resp_message.tool_calls:
            tool_call = resp_message.tool_calls[0]
            print(f"接收到军师指令，准备运行: {tool_call.function.name}")

            # 把拿到的那段文字结果赋值给 tool_result 变量。
            target_method = getattr(self, tool_call.function.name)

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

            tool_result = target_method(**args_dict)

            # 3. 按照标准格式，把执行结果打包
            self.context.append(resp_message) # 必须把军师的指令也存入历史记录
            self.context.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": tool_result # 把你的文字结果交给军师
            })

            # 4. 第二次通信：带着结果回去要最终回复
            response = self.client.chat.completions.create(
                model=self.cur_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant.中文回复"},
                    *self.context,
                ],
                extra_body={"reasoning_split": True},
                tools=self.tools,
                tool_choice="auto"
            )
            resp_message = response.choices[0].message
            self.context.append({"role": "assistant", "content": resp_message.content})
        return response

    def look_at_screen(self):
        try:
            #咔嚓！截取全屏
            screenshot = self.vision.sudden_view()
            

            screen_base64 = pil_image_to_base64(screenshot)
            
            img_msg = pack_msg("user", "image_url", f"data:image/png;base64,{screen_base64}")
            
            self.context.append(img_msg)

            print(f"[本地小弟] 图像base64：\n{screen_base64[:100]}...") # 打印前100个字看看
            return "已生成观察图片message并加入上下文。"
            
        except Exception as e:
            print(f"[本地小弟] 识别失败：{e}")
            return "糟糕，本堂主的眼睛出了点问题，看不清屏幕了。"

def pack_msg(role, type, content):
    if type == "text":
        return {"role": role, "content": content}
    elif type == "image_url":
        return {"role": role, "content": [{"type": "image_url", "image_url": {"url": content}}]}


if __name__ == "__main__":
    test = True

    app = QApplication(sys.argv)
    hutao = Hutao()
    hutao.show()
    sys.exit(app.exec_())