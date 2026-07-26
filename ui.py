import asyncio
from pathlib import Path

import qasync
from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from brain import Brain, pack_msg, parse_tool_args
from listen import Listen
from vision import Vision


class Hutao(QWidget):
    def __init__(self):
        super().__init__()
        """
        中枢神经系统：负责宏观调控
        """

        # UI部分
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer_trick)
        self.timer.start(10000)  # 10秒触发一次

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置透明背景

        self.label = QLabel(self)
        pixmap = QPixmap(str(Path(__file__).parent / "assets" / "hutao.jpg"))
        pixmap = pixmap.scaledToWidth(150, Qt.SmoothTransformation)
        self.label.setPixmap(pixmap)

        # 气泡：显示回复文本，平时隐藏
        self.bubble = QLabel(self)
        self.bubble.setWordWrap(True)
        self.bubble.setMaximumWidth(280)
        self.bubble.setStyleSheet(
            "QLabel { background-color: rgba(255, 255, 255, 230);"
            " border: 2px solid #e88; border-radius: 10px; padding: 8px; }"
        )
        self.bubble.hide()

        # 气泡自动隐藏定时器
        self.bubble_timer = QTimer(self)
        self.bubble_timer.setSingleShot(True)
        self.bubble_timer.timeout.connect(self.hide_bubble)

        # 垂直布局：气泡在上，贴图在下；气泡隐藏时窗口收缩到贴图大小
        layout = QVBoxLayout(self)
        layout.addWidget(self.bubble, alignment=Qt.AlignHCenter)
        layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.adjustSize()

        self.drag_poision = QPoint()

        # 器官部分
        self.brain = Brain()
        self.vision = Vision()
        self.listen = Listen()

        # 连接听觉信号到处理函数
        self.listen.text_signal.connect(self.on_heard_text)  # 发射器.信号.connect(接收器)

        # 自锁
        self.is_busy = False

        # 消息队列-》使用生产者-消费者结构实现
        # 带上限：积压超过 20 条时丢弃新消息，防止无限积压
        self.message_queue = asyncio.Queue(maxsize=20)
        # 创建一个后台任务，专门负责消费消息队列里的消息
        asyncio.get_event_loop().create_task(self.on_received_message_consumer())

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
        self.show_bubble(resp_message.content)

    def show_bubble(self, text, timeout_ms=10000):
        """显示气泡，timeout_ms 后自动隐藏。"""
        self.bubble.setText(text)
        self.bubble.show()
        self.adjustSize()
        self.bubble_timer.start(timeout_ms)

    def hide_bubble(self):
        self.bubble.hide()
        self.bubble_timer.stop()
        self.adjustSize()

    async def tool_executer(self, tool_call):
        # target_method = getattr(self, tool_call.function.name)
        func_name = tool_call.function.name
        _args_dict = parse_tool_args(tool_call.function.arguments)  # 目前工具都无参数，解析以备后续扩展

        if func_name == "look_at_screen":
            img_msg = await self.vision.look_at_screen()
            self.brain.context.append(img_msg)
            return "已查看屏幕并将图片信息加入上下文了哦"

    async def should_reply(self, message):
        try:
            judge_msg = pack_msg(
                "system",
                "text",
                "你是一个聪明的助手，负责判断用户的消息是否需要回复。"
                "如果需要回复，回复true；如果不需要回复，回复false。",
            )
            user_msg = pack_msg("user", "text", f"用户的消息是：{message}")
            judge_context = [judge_msg, user_msg]

            response = await self.brain.get_response_with_context(judge_context)

            reply_decision = response.choices[0].message.content.strip().lower()
            return reply_decision == "true"
        except Exception as e:
            print(f"判断是否回复时出错了：{e}")
            return False

    # asyncSlot：把协程函数当成 Qt 槽函数用。普通 PyQt 只认识同步槽函数，不会自动 await 协程，
    # qasync 的这个装饰器会把协程正确地挂到事件循环里执行。(str) 表示槽函数接收一个 str 类型的信号参数。
    @qasync.asyncSlot(str)
    async def on_heard_text(self, text):
        print(f"接收到听觉消息：{text}")
        try:
            self.message_queue.put_nowait(text)  # 忙碌时也入队，等消费者空闲后处理
            self.show_bubble("听到了，正在想…", timeout_ms=60000)
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    # async def on_received_message_producer(self, message):
    #     self.messsage_queue.put(message)

    async def on_received_message_consumer(self):
        while True:
            message = await self.message_queue.get()  # 当队列为空时就会永远停留在这一行
            # 把忙碌期间积压的消息一并取出合并（多为连续的语音片段）
            pending = [message]
            while not self.message_queue.empty():
                pending.append(self.message_queue.get_nowait())
            if len(pending) > 1:
                message = "\n".join(pending)
                print(f"合并了 {len(pending)} 条积压消息")
            self.is_busy = True
            try:
                print(f"正在处理消息：{message}")
                if await self.should_reply(message):
                    print("判断需要回复，正在处理消息...")
                    await self.do_response(message)
                    # print(response.choices[0].message.content)
                else:
                    print("判断不需要回复，跳过这条消息。")
                    self.hide_bubble()
            except Exception as e:
                print(f"处理消息时出错了：{e}")
                self.hide_bubble()
            finally:
                self.is_busy = False
                # self.message_queue.task_done()  # 和 join 成对出现。当前队列没有调用 join，暂不启用。

    def on_timer_trick(self):
        # resp = self.get_response("请你调用工具看看我的屏幕")
        # print(resp.choices[0].message.content)
        self.vision.sudden_view()

    def mouseDoubleClickEvent(self, event):  # 鼠标双击时
        if event.button() == Qt.LeftButton:
            print("接收到鼠标双击事件")
            self.message_queue.put_nowait("用户用鼠标触碰了你")
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_poision = event.globalPos() - self.frameGeometry().topLeft()  # 以左上角为偏移点
            event.accept()

    def mouseMoveEvent(self, event):  # 鼠标拖动时
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_poision)
            event.accept()
