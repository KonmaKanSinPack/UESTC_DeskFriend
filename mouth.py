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
import time
from collections import deque

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from tts import create_tts

# 朗读播完后的扬声器余响尾巴（秒）：期间麦克风拾音仍视为回声
TTS_ECHO_TAIL = 0.3
# AEC 参考缓冲：播放 PCM 供耳做回声消除（far-end 参考），块长对齐麦克风 32ms/16k
AEC_REF_KEEP = 2.0  # 参考保留时长（秒），防无界增长
AEC_REF_CHUNK = 512  # 块长（16kHz × 32ms，与麦克风块对齐）
AEC_REF_DELAY = 0.15  # 播放→麦克风拾回路径延迟（秒），drain 时按此对齐


def _resample_to_16k(audio, sr):
    """线性重采样到 16k（参考信号够用：AEC 只关心波形大致对应）。"""
    if sr == 16000:
        return np.asarray(audio, dtype=np.float32)
    n = round(len(audio) * 16000 / sr)
    x_new = np.linspace(0, len(audio) - 1, n)
    return np.interp(x_new, np.arange(len(audio)), np.asarray(audio, dtype=np.float32)).astype(np.float32)


class Mouth(QObject):
    """发声器官：speak / interrupt / busy / played_text / speaking 门控。"""

    finished = pyqtSignal()  # 朗读结束（自然播完或被打断均触发）

    def __init__(self, config=None, tts=None):
        """tts 可注入（测试用）；生产路径从 config.toml 经工厂创建（照 Brain 模式）。"""
        super().__init__()
        self.tts = tts or create_tts(config or {})
        self.speaking = False  # 回声门控：耳朵读它忽略扬声器回声（含余响尾巴）
        self.speak_started_at = 0.0  # 本次朗读开始时刻（耳判断 AEC 收敛期）
        self._ref_buf = deque()  # AEC 参考缓冲：(播放时刻, 512块 float32 16k)
        if hasattr(self.tts, "ref_callback"):  # 后端支持播放上报（SiliconFlow/CosyVoice2）
            self.tts.ref_callback = self._tee_reference

    async def speak(self, text):
        """命令：朗读文本。

        门控置位覆盖合成+播放全段；结束后延迟余响尾巴再释放（try/finally 保证
        被打字打断/后端异常路径也释放，防止门控卡死导致一直忽略用户语音）。
        """
        if not text:
            return
        self.speak_started_at = time.monotonic()
        self.speaking = True
        try:
            await self.tts.speak(text)
        finally:
            await asyncio.sleep(TTS_ECHO_TAIL)
            self.speaking = False
            self._ref_buf.clear()  # 清参考：防止陈旧参考被当成回声去消
            self.finished.emit()

    def _tee_reference(self, audio, sr, t_play=None):
        """播放上报：重采样到 16k，按 512 块切分（尾块补零）带播放时刻入参考缓冲。

        t_play 由后端在 sd.play 前一刻记录（实际播放时刻，比 tee 时刻准——
        每句新建流的启动延迟 70~170ms 逐句变化，用 tee 时刻对齐误差可达 ±100ms）。
        """
        a16 = _resample_to_16k(audio, sr)
        now = time.monotonic()
        base = t_play if t_play is not None else now
        for k in range(0, len(a16), AEC_REF_CHUNK):
            block = a16[k : k + AEC_REF_CHUNK]
            if len(block) < AEC_REF_CHUNK:
                block = np.pad(block, (0, AEC_REF_CHUNK - len(block)))
            self._ref_buf.append((base + k / 16000.0, block))
        cutoff = now - AEC_REF_KEEP  # 裁剪超龄条目（防无界增长）
        while self._ref_buf and self._ref_buf[0][0] < cutoff:
            self._ref_buf.popleft()

    def drain_reference(self, now=None, delay=AEC_REF_DELAY):
        """取播放时刻 ≤ now-delay 的块作当前麦克风块的 far-end 参考；无则 None。

        耳每 32ms 调一次；嘴未发声时缓冲为空 → None（AEC 不消，无回声可消）。
        """
        if now is None:
            now = time.monotonic()
        ref = None
        while self._ref_buf and self._ref_buf[0][0] <= now - delay:
            ref = self._ref_buf.popleft()
        return ref[1] if ref is not None else None

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
