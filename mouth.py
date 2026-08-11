"""语音输出（嘴）器官：与 listen.py（耳）对称的纯物理层。

职责：说话。不知道大模型存在——speak(text) 是主控中心（ui/Hutao）的命令，
interrupt() 返回的打断位置由主控中心决定注入哪里（brain.set_interruption）。

打断设计（句粒度监听窗口，2026-08-12 替代 AEC barge-in）：
- 句子播放中：耳完全不监听（speaking=True 且 window_open=False → 耳 drop）
- 句与句之间：监听窗口（guard 余响静默 → window 耳拾音）；窗口内 VAD 触发
  = 用户说话 → 主控立即打断（interrupt_requested 信号）
- 确定性高于 AEC：无收敛期/对齐/参考缓冲问题，代价是播放中插嘴要等当前句
  读完、窗口期才被听到

内部消化：TTS 后端装配（create_tts 工厂）、speaking 门控、句间监听窗口、
try/finally 保证门控释放；finished 信号广播朗读结束。
"""

import asyncio
import time

from PyQt5.QtCore import QObject, pyqtSignal

from tts import create_tts

# 朗读播完后的扬声器余响尾巴（秒）：期间麦克风拾音仍视为回声
TTS_ECHO_TAIL = 0.3
# 句间监听窗口（秒）：guard 为上一句余响静默期（耳仍 drop），window 为拾音期
SENTENCE_GUARD = 0.35  # 余响静默：上一句结尾强音反射未散，此时拾音必误触发
SENTENCE_WINDOW = 0.6  # 监听窗口：用户在此间说话 → 打断（句粒度进度保留）


class Mouth(QObject):
    """发声器官：speak / interrupt / busy / played_text / speaking 门控 / 监听窗口。"""

    finished = pyqtSignal()  # 朗读结束（自然播完或被打断均触发）

    def __init__(self, config=None, tts=None):
        """tts 可注入（测试用）；生产路径从 config.toml 经工厂创建（照 Brain 模式）。"""
        super().__init__()
        self.tts = tts or create_tts(config or {})
        self.speaking = False  # 门控：朗读会话中（含尾巴）；耳据此 drop/监听
        self.window_open = False  # 句间监听窗口开：耳只在窗口期拾音
        if hasattr(self.tts, "sentence_done_callback"):  # 后端支持句间钩子
            self.tts.sentence_done_callback = self._sentence_gap

    async def speak(self, text):
        """命令：朗读文本。

        门控置位覆盖合成+播放全段；结束后延迟余响尾巴再释放（try/finally 保证
        被打字打断/后端异常路径也释放，防止门控卡死导致一直忽略用户语音）。
        """
        if not text:
            return
        self.speaking = True
        try:
            await self.tts.speak(text)
        finally:
            await asyncio.sleep(TTS_ECHO_TAIL)
            self.speaking = False
            self.finished.emit()

    def _sentence_gap(self):
        """句间监听窗口：guard 余响静默 → 打开窗口 → 关闭。

        由后端在每句播完后（非末句）调用，阻塞在 speak 线程 = 播放暂停；
        窗口内被打断（stop_event 置位）→ 后端循环顶部 break，句粒度终止。
        """
        self.window_open = False
        time.sleep(SENTENCE_GUARD)  # 余响静默期：耳仍 drop
        self.window_open = True
        time.sleep(SENTENCE_WINDOW)  # 监听窗口：耳拾音，VAD → 打断
        self.window_open = False

    async def interrupt(self) -> str:
        """命令：打断当前朗读，返回已播放文本前缀（打断位置）。

        注入对话上下文由主控中心（ui）决定，器官不依赖 brain。
        """
        return await self.tts.interrupt()

    @property
    def busy(self) -> bool:
        """后端是否正在朗读（不含尾巴与句间窗口；窗口期 busy=False 但 speaking=True）。"""
        return self.tts.busy

    @property
    def played_text(self) -> str:
        """当前（或最近一次）朗读已播放文本前缀；无播放返回空串。"""
        return self.tts.played_text
