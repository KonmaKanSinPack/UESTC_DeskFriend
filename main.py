import asyncio
import signal
import sys

# Windows 下必须先加载 onnxruntime 再加载 PyQt5，否则 Qt 的 DLL 会遮蔽
# onnxruntime 依赖的同名 DLL，导致 onnxruntime_pybind11_state 导入失败
import onnxruntime  # noqa: F401
import qasync
from PyQt5.QtWidgets import QApplication

from skin import Skin
from spine import Spine

if __name__ == "__main__":
    # Qt 事件循环不返回 Python 解释器，SIGINT 的 Python 处理器永远得不到执行，
    # 表现为 Ctrl+C 无法退出。改用默认动作，让内核直接终止进程。
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app)  # 创建兼容PyQt的异步事件
    asyncio.set_event_loop(loop)  # 设置异步事件循环

    face = Skin()  # 外观器官（窗口本体）
    spine = Spine(face=face)  # 主控中枢：内部装配 brain/vision/listen/mouth 并接线
    spine.start()  # 起消费者任务（loop 已 set 未 run，走 get_event_loop 回退分支）
    face.show()

    with loop:
        loop.run_forever()
