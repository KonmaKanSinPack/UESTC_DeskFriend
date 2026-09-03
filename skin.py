"""外观（皮）器官：与 mouth.py（嘴，发声）对称的纯物理层。

职责：显示（贴图/气泡/动画/托盘）+ 触摸输入（点击/拖动/双击/右键/文字输入）。
不知道大模型存在——show_bubble/set_anim_state 是主控中心（spine）的命令；
text_submitted/touched 是向主控广播的信号（器官永远广播，主控决定用不用）。

未来 Live2D 演进：渲染方式（SpriteRenderer/Live2DRenderer）是皮内部细节；
set_anim_state 是语义级命令（"表达正在思考"），为渲染器替换留好缝。
"""

import math
from pathlib import Path

import tomllib
from PyQt5.QtCore import QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QMovie, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QMenu, QStyle, QSystemTrayIcon, QVBoxLayout, QWidget

PROJECT_DIR = Path(__file__).parent
SPRITE_WIDTH = 150  # 贴图统一缩放到这个宽度
# 动画帧率分档（2026-09-03 流畅性轮）：idle 呼吸是慢正弦，8fps 视觉无差、CPU 降 ~2/3
# （动整窗的分层窗口每帧都触发 DWM 全窗 alpha 重合成，是最贵的动画路径——帧率即电费）；
# thinking 晃动 / talking 弹跳动作快，保 25fps
ANIM_INTERVAL_IDLE = 125  # ms ≈ 8fps
ANIM_INTERVAL_ACTIVE = 40  # ms = 25fps
# 相位步进按毫秒等比：原实现 0.12/帧 @40ms，换帧率后保持同一节奏（否则 idle 下呼吸会变慢 3 倍）
ANIM_PHASE_PER_MS = 0.12 / 40


def load_config():
    with open(PROJECT_DIR / "config.toml", "rb") as f:
        return tomllib.load(f)


class Skin(QWidget):
    """外观器官：贴图 + 气泡 + 输入框 + 状态动画 + 拖动/单击/双击输入。"""

    # —— 信号（广播输入；skin 不知道谁会来听）——
    text_submitted = pyqtSignal(str)  # 回车发送的文字（空文本 strip 后不发）
    touched = pyqtSignal()  # 双击"触碰"彩蛋
    quit_requested = pyqtSignal()  # 托盘/右键菜单「退出」：只广播，收摊编排归主控

    def __init__(self):
        super().__init__()

        # Qt.Tool：常驻桌宠不进任务栏与 Alt-Tab（无它会在任务栏占一个常驻位）
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置透明背景

        self.label = QLabel(self)
        self._load_sprite()
        self._init_tray()

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
        self.input_box.returnPressed.connect(self._on_input_submitted)
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
        self.anim_timer.start(ANIM_INTERVAL_IDLE)  # 起始即 idle 档

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

    def _init_tray(self):
        """系统托盘：桌宠的常驻退出入口（Qt.Tool 后任务栏不再有窗口按钮，托盘成为主入口）。

        原理：QSystemTrayIcon 把图标注册进系统托盘区，setContextMenu 后系统在
        图标右键时弹出菜单。菜单项的 triggered 信号会带 checked 参数，而本类广播的
        quit_requested 是无参信号——直连会 TypeError，所以用 lambda 丢弃参数再 emit
        （复现：以后加「静音 TTS」等菜单项，照抄这条 connect 写法即可）。
        isSystemTrayAvailable() 为假（精简 Linux / 无壳环境）时静默跳过，不报错——
        桌宠右键菜单仍是等价退出兜底，两条入口广播同一个信号。
        """
        self.tray = QSystemTrayIcon(self._tray_icon(), self)
        self.tray.setToolTip("桃桃 (UESTC_DeskFriend)")
        menu = QMenu(self)  # 挂父对象：没有引用的 QMenu 会被 Python GC 回收，菜单弹出即消失
        menu.addAction("退出").triggered.connect(lambda: self.quit_requested.emit())
        self.tray.setContextMenu(menu)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray.show()

    def _tray_icon(self):
        """托盘图标三级回退：GIF 当前帧 → 静态贴图 → 系统占位图标。

        QIcon 持有各尺寸位图、由系统自选（Windows 托盘约 16×16），无需手动缩放；
        只在两极都拿不到图（贴图路径配错）时才落到 SP_ComputerIcon，保证托盘永不缺位。
        """
        movie = getattr(self, "movie", None)  # GIF 分支才有 movie 属性
        if movie is not None:
            return QIcon(QPixmap.fromImage(movie.currentImage()))
        pixmap = self.label.pixmap()
        if pixmap is not None and not pixmap.isNull():
            return QIcon(pixmap)
        return QApplication.style().standardIcon(QStyle.SP_ComputerIcon)

    # ---------- 命令（被主控命令；skin 不知道命令来自谁） ----------

    def show_bubble(self, text, timeout_ms=10000):
        """显示气泡，timeout_ms 后自动隐藏（隐藏时自动复位动画状态）。"""
        self.bubble.setText(text)
        self.bubble.show()
        self.adjustSize()
        self.bubble_timer.start(timeout_ms)

    def hide_bubble(self):
        self.bubble.hide()
        self.bubble_timer.stop()
        self.adjustSize()
        self._apply_anim_state("idle")

    def set_anim_state(self, state):
        """语义级命令："idle" 待机呼吸 / "thinking" 思考晃动 / "talking" 说话弹跳。

        渲染方式（整窗微动 / 未来 Live2D）是皮肤内部细节。
        """
        self._apply_anim_state(state)

    def _apply_anim_state(self, state):
        """状态即帧率档：帧率跟状态一起切（QTimer.setInterval 不重置计时，安全）。

        原理/复现：idle 呼吸是周期 ~2s 的慢正弦，8fps 采样视觉无差；换挡的依据
        是"动作频率"——晃动/弹跳的位移变化快，降帧会肉眼可见卡顿，故保 25fps。
        """
        self._anim_state = state
        self.anim_timer.setInterval(ANIM_INTERVAL_IDLE if state == "idle" else ANIM_INTERVAL_ACTIVE)

    # ---------- 内部：外观细节自管理 ----------

    def _on_input_submitted(self):
        """回车发送：校验后广播 text_submitted（入队/打断等编排归主控）。"""
        text = self.input_box.text().strip()
        self.input_box.clear()
        self.input_box.hide()
        self.adjustSize()
        if not text:
            return
        self.text_submitted.emit(text)

    def _anim_tick(self):
        """整窗微动动画。拖动中不动，避免和用户抢窗口位置。"""
        if self._dragging:
            return
        if self._base_pos is None:
            self._base_pos = self.pos()
            return
        # 相位按真实毫秒推进（0.003/ms）：帧率分档后各档节奏一致（呼吸周期不变）
        self._anim_phase += ANIM_PHASE_PER_MS * self.anim_timer.interval()
        if self._anim_state == "thinking":
            offset = QPoint(round(3 * math.sin(self._anim_phase * 4)), 0)  # 快速晃动
        elif self._anim_state == "talking":
            offset = QPoint(0, -abs(round(4 * math.sin(self._anim_phase * 2))))  # 弹跳
        else:
            offset = QPoint(0, round(2 * math.sin(self._anim_phase)))  # 缓慢呼吸
        self.move(self._base_pos + offset)

    def eventFilter(self, obj, event):
        # 输入框里按 Esc 收起
        if obj is self.input_box and event.type() == event.KeyPress and event.key() == Qt.Key_Escape:
            self.input_box.hide()
            self.adjustSize()
            return True
        return super().eventFilter(obj, event)

    def contextMenuEvent(self, event):
        """右键菜单：与托盘同款退出入口，同样只广播信号。

        原理：Windows 上右键释放时 Qt 自动合成 ContextMenu 事件（与 mousePressEvent
        里只处理 LeftButton 的分支互不干扰，无需在 mouse 事件里判右键）。menu.exec_
        以模态方式运行到用户选中一项或点击别处才返回，期间不占 CPU。
        """
        menu = QMenu(self)
        menu.addAction("退出").triggered.connect(lambda: self.quit_requested.emit())
        menu.exec_(event.globalPos())
        event.accept()

    def mouseDoubleClickEvent(self, event):  # 鼠标双击时
        if event.button() == Qt.LeftButton:
            print("接收到鼠标双击事件")
            self._just_double_clicked = True  # 防止双击被误判成两次单击
            self.touched.emit()  # 广播触碰；入队什么由主控决定
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
