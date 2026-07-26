import asyncio
import signal
import sys

import qasync
from PyQt5.QtWidgets import QApplication

from ui import Hutao

if __name__ == "__main__":
    # Qt 事件循环不返回 Python 解释器，SIGINT 的 Python 处理器永远得不到执行，
    # 表现为 Ctrl+C 无法退出。改用默认动作，让内核直接终止进程。
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)

    loop = qasync.QEventLoop(app)  # 创建兼容PyQt的异步事件
    asyncio.set_event_loop(loop)  # 设置异步事件循环

    hutao = Hutao()
    hutao.show()

    with loop:
        loop.run_forever()
