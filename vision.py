import base64
import os
import shutil
import subprocess
import tempfile
from collections import deque
from io import BytesIO

from PIL import Image, ImageGrab

from brain import pack_msg


def pil_image_to_base64(pil_image):
    buffered = BytesIO()  # 制造一个存在于内存里的“虚拟文件”
    pil_image.save(buffered, format="PNG")  # 把内存里的图片对象，存进这个虚拟文件里（指定格式为 PNG）

    # 提取虚拟文件里的二进制数据，打包成 base64 文本
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_image_to_base64_by_path(image_path):
    # 以二进制读取模式("rb")打开图片，并转换为 base64 文本
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


_pw_capture = "uninit"  # PipeWire 后端惰性单例："uninit" 未碰过 / None 不可用 / 实例


def _get_pw_capture():
    """拿到 PipeWire 静默截屏后端；仅 Wayland 且系统装有 gi/GStreamer 时可用。"""
    global _pw_capture
    if _pw_capture != "uninit":
        return _pw_capture
    _pw_capture = None
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        try:
            from pw_capture import PipeWireCapture

            _pw_capture = PipeWireCapture()
        except Exception:
            pass  # 缺 gi / GStreamer 时安静回退到 gnome-screenshot
    return _pw_capture


def grab_screenshot():
    """截一张全屏图，返回 PIL.Image。

    Wayland 会话下优先走 PipeWire 后端（ScreenCast 门户授权一次后完全静默）；
    后端未就绪或抓帧失败时回退 gnome-screenshot 子进程（有闪光灯效），
    因为 GNOME Shell 的私有 Screenshot D-Bus 接口对第三方返回 AccessDenied；
    X11 会话直接用 ImageGrab。
    """
    is_wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    if is_wayland:
        backend = _get_pw_capture()
        if backend is not None:
            image = backend.grab()
            if image is not None:
                return image
    if is_wayland and shutil.which("gnome-screenshot"):
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            subprocess.run(
                ["gnome-screenshot", "-f", tmp.name],
                check=True,
                capture_output=True,
                timeout=10,
            )
            with Image.open(tmp.name) as img:
                return img.copy()
        finally:
            os.unlink(tmp.name)
    return ImageGrab.grab()


class Vision:  # AI的视觉模块
    def __init__(self, history_length=5):
        self.history = deque(maxlen=history_length)

    def update(self, screenshot):
        self.history.append(screenshot)

    def sudden_view(self):
        self.update(grab_screenshot().resize((224, 224)))
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
