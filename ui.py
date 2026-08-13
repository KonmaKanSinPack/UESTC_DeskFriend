import asyncio
import math
from pathlib import Path

import qasync
import tomllib
from PyQt5.QtCore import QPoint, QSize, Qt, QTimer
from PyQt5.QtGui import QMovie, QPixmap
from PyQt5.QtWidgets import QLabel, QLineEdit, QVBoxLayout, QWidget

from backends.judger import JUDGE_SYSTEM_PROMPT
from brain import Brain, pack_msg, parse_tool_args
from listen import Listen
from mouth import Mouth
from vision import Vision

PROJECT_DIR = Path(__file__).parent
SPRITE_WIDTH = 150  # 贴图统一缩放到这个宽度


def load_config():
    with open(PROJECT_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)


class DeskFriend(QWidget):
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
        self._load_sprite()

        # 气泡：显示回复文本，平时隐藏
        self.bubble = QLabel(self)
        self.bubble.setWordWrap(True)
        self.bubble.setMaximumWidth(280)
        self.bubble.setStyleSheet(
            "QLabel { background-color: rgba(255, 255, 255, 230);"
            " border: 2px solid #e88; border-radius: 10px; padding: 8px; }"
        )
        self.bubble.hide()

        # 文字输入框：单击桌宠唤起，回车发送，Esc 收起
        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("和桃桃说点什么…")
        self.input_box.setMaximumWidth(280)
        self.input_box.hide()
        self.input_box.returnPressed.connect(self.on_input_submitted)
        self.input_box.installEventFilter(self)

        # 气泡自动隐藏定时器
        self.bubble_timer = QTimer(self)
        self.bubble_timer.setSingleShot(True)
        self.bubble_timer.timeout.connect(self.hide_bubble)

        # 垂直布局：气泡、输入框在上，贴图在下；都隐藏时窗口收缩到贴图大小
        layout = QVBoxLayout(self)
        layout.addWidget(self.bubble, alignment=Qt.AlignHCenter)
        layout.addWidget(self.input_box, alignment=Qt.AlignHCenter)
        layout.addWidget(self.label, alignment=Qt.AlignHCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        self.adjustSize()

        self.drag_poision = QPoint()

        # 程序动画：整窗微动模拟待机呼吸/思考晃动/说话弹跳
        self._anim_state = "idle"  # idle / thinking / talking
        self._anim_phase = 0.0
        self._base_pos = None  # 动画的基准位置（拖动结束后更新）
        self._dragging = False
        self._press_pos = QPoint()  # 按下时的全局坐标，用于区分单击与拖动
        self._just_double_clicked = False
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_timer.start(40)  # 25fps

        # 器官部分
        self.brain = Brain()
        # 主动冒泡回调：astrbot 后端（屏幕感知）发现值得说的话时直接显示气泡
        self.brain.reply_sink = self.show_bubble
        self.vision = Vision()
        self.listen = Listen()
        self.mouth = Mouth(load_config())  # 发声器官（嘴）：朗读 + 打断 + 回声门控
        self.listen.mouth = self.mouth  # 耳朵引用嘴：朗读期间忽略扬声器回声
        # 插嘴打断：句间监听窗口内检测到用户说话 → 打断（复用现有注入链路）
        self.listen.interrupt_requested.connect(self._interrupt_tts)

        # 连接听觉信号到处理函数
        self.listen.text_signal.connect(self.on_heard_text)  # 发射器.信号.connect(接收器)

        # 自锁
        self.is_busy = False

        # 消息队列-》使用生产者-消费者结构实现
        # 带上限：积压超过 20 条时丢弃新消息，防止无限积压
        self.message_queue = asyncio.Queue(maxsize=20)
        # 创建一个后台任务，专门负责消费消息队列里的消息
        asyncio.get_event_loop().create_task(self.on_received_message_consumer())

    def _load_sprite(self):
        """从 config.toml 的 SPRITE 加载贴图，支持静态图与 GIF 动图。"""
        sprite = load_config().get("SPRITE", "assets/nuonuo.png")
        sprite_path = Path(sprite)
        if not sprite_path.is_absolute():
            sprite_path = PROJECT_DIR / sprite_path

        if sprite_path.suffix.lower() == ".gif":
            self.movie = QMovie(str(sprite_path))
            self.movie.jumpToFrame(0)  # 先取一帧拿到原始尺寸，按比例算缩放
            frame_size = self.movie.currentImage().size()
            scaled = QSize(SPRITE_WIDTH, round(SPRITE_WIDTH * frame_size.height() / frame_size.width()))
            self.movie.setScaledSize(scaled)
            self.label.setMovie(self.movie)
            self.movie.start()
        else:
            pixmap = QPixmap(str(sprite_path))
            pixmap = pixmap.scaledToWidth(SPRITE_WIDTH, Qt.SmoothTransformation)
            self.label.setPixmap(pixmap)

    async def do_response(self, message):
        response = await self.brain.get_llm_response(message)
        while response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"接收到军师指令，准备运行: {tool_call.name}")
            tool_result, extra_msg = await self.tool_executer(tool_call)

            # 按照标准格式，把执行结果打包；tool 消息必须紧跟 assistant 的 tool_calls，
            # 截图等附加消息放在 tool 之后，否则 API 判定 role 序列非法返回 400
            tool_msg = pack_msg("tool", "tool", tool_result, tool_call)
            msgs = [tool_msg] + ([extra_msg] if extra_msg else [])

            # 第二次通信：带着结果回去要最终回复
            response = await self.brain.get_llm_response(msgs)

        print(response.content)
        self._anim_state = "talking"
        self.show_bubble(response.content)
        # 桥超时/断线兜底（answered=False）：气泡给反馈，但不朗读——否则兜底文案被
        # 扬声器放出→麦克风拾回→回声自回复环（详见 docs/session/2026-08-13.md 组B）
        if not response.answered:
            return
        # 朗读回复（异步不阻塞对话队列；门控置位/尾巴释放由 Mouth 内部消化）
        asyncio.get_event_loop().create_task(self.mouth.speak(response.content))
        # 回复已展示，再后台做记忆压缩（超限时把最老轮次并入摘要，失败不影响对话）
        await self.brain.maybe_compress()
        # 批量抽取自上次以来的事实（含此前攒下的背景谈话，失败不影响对话）
        await self.brain.maybe_extract_facts()

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
        self._anim_state = "idle"

    def _anim_tick(self):
        """整窗微动动画。拖动中不动，避免和用户抢窗口位置。"""
        if self._dragging:
            return
        if self._base_pos is None:
            self._base_pos = self.pos()
            return
        self._anim_phase += 0.12
        if self._anim_state == "thinking":
            offset = QPoint(round(3 * math.sin(self._anim_phase * 4)), 0)  # 快速晃动
        elif self._anim_state == "talking":
            offset = QPoint(0, -abs(round(4 * math.sin(self._anim_phase * 2))))  # 弹跳
        else:
            offset = QPoint(0, round(2 * math.sin(self._anim_phase)))  # 缓慢呼吸
        self.move(self._base_pos + offset)

    def _interrupt_tts(self):
        """用户新消息到达：打断朗读并记录打断位置（异步不阻塞入队）。"""
        if self.mouth.busy:
            asyncio.get_event_loop().create_task(self._interrupt_tts_async())

    async def _interrupt_tts_async(self):
        # 主控中心编排：嘴只返回打断位置，注入对话上下文由这里决定
        prefix = await self.mouth.interrupt()
        if prefix:
            self.brain.set_interruption(prefix)
            print(f"朗读被打断，记录位置：…{prefix[-15:]}")

    def on_input_submitted(self):
        text = self.input_box.text().strip()
        self.input_box.clear()
        self.input_box.hide()
        self.adjustSize()
        if not text:
            return
        print(f"接收到文字消息：{text}")
        self._interrupt_tts()  # 打断朗读，记录打断位置
        try:
            self.message_queue.put_nowait(text)
            self.show_bubble("听到了，正在想…", timeout_ms=60000)
            self._anim_state = "thinking"
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def eventFilter(self, obj, event):
        # 输入框里按 Esc 收起
        if obj is self.input_box and event.type() == event.KeyPress and event.key() == Qt.Key_Escape:
            self.input_box.hide()
            self.adjustSize()
            return True
        return super().eventFilter(obj, event)

    async def tool_executer(self, tool_call):
        # tool_call 是统一 ToolCall 对象（openai 后端归一化而来；astrbot 后端不产生工具调用）
        func_name = tool_call.name
        _args_dict = parse_tool_args(tool_call.arguments)  # 目前工具都无参数，解析以备后续扩展

        if func_name == "look_at_screen":
            result = await self.vision.look_at_screen()
            if isinstance(result, dict):
                # 成功：图片消息由 do_response 按序插入 context（tool 消息之后）
                return "已查看屏幕并将图片信息加入上下文了哦", result
            # 失败/未就绪：返回的是提示文本，直接作为工具结果
            return result, None

    async def should_reply(self, message):
        try:
            # 判定提示词单一来源：backends/judger.py（两个后端共用，勿在此内联）
            judge_msg = pack_msg("system", "text", JUDGE_SYSTEM_PROMPT)
            user_msg = pack_msg("user", "text", f"用户的消息是：{message}")
            judge_context = [judge_msg, user_msg]

            response = await self.brain.get_response_with_context(judge_context)

            reply_decision = response.content.strip().lower()
            print(f"查看决策器纯净输出：{reply_decision}")
            return reply_decision == "true"
        except Exception as e:
            print(f"判断是否回复时出错了：{e}")
            return False

    # asyncSlot：把协程函数当成 Qt 槽函数用。普通 PyQt 只认识同步槽函数，不会自动 await 协程，
    # qasync 的这个装饰器会把协程正确地挂到事件循环里执行。(str) 表示槽函数接收一个 str 类型的信号参数。
    @qasync.asyncSlot(str)
    async def on_heard_text(self, text):
        print(f"接收到听觉消息：{text}")
        if self.mouth.is_echo(text):
            # 播放结束后余响（~1.3s）转写出的就是桃桃刚说的话：内容兜底，丢弃
            print(f"与刚朗读内容相似（疑似回声），丢弃：{text}")
            return
        self._interrupt_tts()  # 打断朗读，记录打断位置
        try:
            self.message_queue.put_nowait(text)  # 忙碌时也入队，等消费者空闲后处理
            self.show_bubble("听到了，正在想…", timeout_ms=60000)
            self._anim_state = "thinking"
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
                    print("判断不需要回复，仅记入记忆。")
                    self.brain.memorize(message)  # 背景谈话只记不答
                    await self.brain.maybe_compress()  # 跳过回复的消息也要参与压缩
                    self.hide_bubble()
            except Exception as e:
                print(f"处理消息时出错了：{e}")
                self.hide_bubble()
            finally:
                self.is_busy = False
                # self.message_queue.task_done()  # 和 join 成对出现。当前队列没有调用 join，暂不启用。

    def on_timer_trick(self):
        # 截屏是阻塞 IO，丢线程池执行，避免每 10s 一次的定时截屏卡住 qasync 事件循环 →
        # 延迟耳线程 emit 的 interrupt_requested 投递（#9，见 docs/session/2026-08-13.md 组E）
        asyncio.get_event_loop().create_task(asyncio.to_thread(self.vision.sudden_view))

    def mouseDoubleClickEvent(self, event):  # 鼠标双击时
        if event.button() == Qt.LeftButton:
            print("接收到鼠标双击事件")
            self._just_double_clicked = True  # 防止双击被误判成两次单击
            self.message_queue.put_nowait("用户用鼠标触碰了你")
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._press_pos = event.globalPos()
            self.drag_poision = event.globalPos() - self.frameGeometry().topLeft()  # 以左上角为偏移点
            event.accept()

    def mouseMoveEvent(self, event):  # 鼠标拖动时
        if self._dragging and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_poision)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._dragging:
            self._dragging = False
            self._base_pos = self.pos()  # 动画基准跟随新位置
            # 几乎没移动视为单击：唤起/收起文字输入框
            if (event.globalPos() - self._press_pos).manhattanLength() < 6:
                if self._just_double_clicked:
                    self._just_double_clicked = False
                elif self.input_box.isVisible():
                    self.input_box.hide()
                    self.adjustSize()
                else:
                    self.input_box.show()
                    self.input_box.setFocus()
                    self.adjustSize()
            event.accept()
