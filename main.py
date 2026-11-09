import win_compat  # noqa: F401  # 必须先于 PyQt5/qasync：预加载系统 MSVC 运行库

import asyncio
import signal
import sys

import qasync
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from ui import DeskFriend

if __name__ == "__main__":
    # Qt 事件循环不返回 Python 解释器，SIGINT 的 Python 处理器永远得不到执行，
    # 表现为 Ctrl+C 无法退出。改用默认动作，让内核直接终止进程。
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    if sys.platform == "win32":
        # Qt5 在 Windows 上默认不启用高 DPI 缩放；开启后贴图、鼠标坐标与截屏分辨率一致
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app)  # 创建兼容PyQt的异步事件
    asyncio.set_event_loop(loop)  # 设置异步事件循环

    pet = DeskFriend()
    pet.show()

    with loop:
        loop.run_forever()
