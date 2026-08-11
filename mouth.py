"""语音输出（嘴）器官：与 listen.py（耳）对称的纯物理层。

职责：说话。不知道大模型存在——speak(text) 是主控中心（ui/Hutao）的命令，
interrupt() 返回的打断位置由主控中心决定注入哪里（brain.set_interruption）。

内部消化：
- TTS 后端装配（create_tts 工厂，照 brain 门面 + backends/ 实现模式；
  tts.py 保留抽象与后端实现，器官主体不被"未完成"的 CosyVoice2 污染）
- 回声门控置位/释放：speaking 覆盖合成+播放全段 + 余响尾巴，
  try/finally 保证被打断/异常路径也释放门控（防"一直忽略用户语音"）
- finished 信号：朗读结束广播（自然播完或被打断），主控中心可监听做收尾

耳朵（listen.py）读 self.speaking 忽略扬声器回声（ui 装配时注入 mouth 引用）。
"""

import asyncio

from PyQt5.QtCore import QObject, pyqtSignal

from tts import create_tts

# 朗读播完后的扬声器余响尾巴（秒）：期间麦克风拾音仍视为回声
TTS_ECHO_TAIL = 0.3


class Mouth(QObject):
    """发声器官：speak / interrupt / busy / played_text / speaking 门控。"""

    finished = pyqtSignal()  # 朗读结束（自然播完或被打断均触发）

    def __init__(self, config=None, tts=None):
        """tts 可注入（测试用）；生产路径从 config.toml 经工厂创建（照 Brain 模式）。"""
        super().__init__()
        self.tts = tts or create_tts(config or {})
        self.speaking = False  # 回声门控：耳朵读它忽略扬声器回声（含余响尾巴）

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

    async def interrupt(self) -> str:
        """命令：打断当前朗读，返回已播放文本前缀（打断位置）。

        注入对话上下文由主控中心（ui）决定，器官不依赖 brain。
        """
        return await self.tts.interrupt()

    @property
    def busy(self) -> bool:
        """后端是否正在朗读（不含余响尾巴；尾巴期间 busy=False 但 speaking=True）。"""
        return self.tts.busy

    @property
    def played_text(self) -> str:
        """当前（或最近一次）朗读已播放文本前缀；无播放返回空串。"""
        return self.tts.played_text
