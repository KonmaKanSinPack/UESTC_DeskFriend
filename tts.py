"""语音输出（TTS）器官：与 listen.py（语音输入）对称。

架构：TTS 抽象 + 可插拔实现，按 config.toml 的 TTS_BACKEND 键装配（照 backends 工厂模式）。
当前实现：
- DummyTTS（占位）：记录调用不打开发声，先跑通「朗读 → 打断 → 打断位置进 context」链路
- CosyVoice2TTS（stub）：本地合成，待环境就绪后接入（torch + 模型 + 参考音频）

打断制（用户设计）：朗读中用户发言 → interrupt() 立即停止 → 返回已播放文本前缀
（打断位置）→ 由上层注入对话上下文，让桃桃知道"说到哪被打断了"。
"""

import asyncio
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTS(ABC):
    """语音输出抽象：合成 + 流式播放 + 打断。实现者负责句粒度进度跟踪。"""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """合成并播放文本；可被 interrupt() 打断。"""

    @abstractmethod
    async def interrupt(self) -> str:
        """打断当前朗读，返回已播放文本前缀（打断位置记号）。

        无播放时返回空串。调用后 busy 为 False。
        """

    @property
    @abstractmethod
    def busy(self) -> bool:
        """是否正在朗读。"""

    @property
    @abstractmethod
    def played_text(self) -> str:
        """当前（或最近一次）朗读中已播放的文本前缀；无播放返回空串。"""

    async def close(self) -> None:
        """释放资源（模型/音频流）；默认空操作。"""


class DummyTTS(TTS):
    """占位实现：记录调用、模拟句粒度播放，不打开发声。

    用于在 CosyVoice2 接入前跑通「朗读 → 打断 → 打断位置注入」全链路；
    句间隔可配（默认 0.01s），测试时置 0 或加大以便观察。
    """

    def __init__(self, sentence_gap: float = 0.01):
        self.sentence_gap = sentence_gap
        self._busy = False
        self._played = ""
        self.speak_calls: list[str] = []  # 测试/日志用

    async def speak(self, text: str) -> None:
        if not text:
            return
        self.speak_calls.append(text)
        self._busy = True
        self._played = ""
        # 按句模拟流式播放（CosyVoice2 也是按句产出 chunk）
        sentences = [s for s in text.replace("\n", "，").split("。") if s]
        try:
            for i, sent in enumerate(sentences):
                self._played += sent + "。"
                if self.sentence_gap > 0:
                    await asyncio.sleep(self.sentence_gap)
                if i == len(sentences) - 1:
                    self._played = text  # 播完 = 完整文本
        except asyncio.CancelledError:
            raise
        finally:
            self._busy = False

    async def interrupt(self) -> str:
        self._busy = False
        return self._played if self._played and self._played != self.speak_calls[-1] else ""

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def played_text(self) -> str:
        return self._played


class CosyVoice2TTS(TTS):
    """CosyVoice2 本地合成（stub，待接入）。

    接入要求：torch+CUDA 环境、模型权重（CosyVoice2-0.5B）、参考音频与文本
    （零样本克隆音色）。当前占位：构造时打印提示，speak 仅记录。
    """

    def __init__(self, model_dir: str = "", voice_ref: str = "", voice_ref_text: str = ""):
        self.model_dir = model_dir
        self.voice_ref = voice_ref
        self.voice_ref_text = voice_ref_text
        self._busy = False
        self._played = ""
        self.speak_calls: list[str] = []
        logger.warning(
            "CosyVoice2 尚未接入（框架占位）：请在 tts.CosyVoice2TTS 实现合成逻辑。参考音频：%s",
            voice_ref or "（未配置）",
        )

    async def speak(self, text: str) -> None:
        self.speak_calls.append(text)
        logger.warning("[CosyVoice2 占位] 应朗读：%s…", text[:20])

    async def interrupt(self) -> str:
        return ""

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def played_text(self) -> str:
        return self._played


def create_tts(config) -> TTS:
    """按配置装配 TTS 后端（照 backends 工厂模式）。"""
    backend = config.get("TTS_BACKEND", "dummy")
    if backend == "none":
        return DummyTTS()  # none 与 dummy 等价：占位链路可用，不发声
    if backend == "cosyvoice2":
        return CosyVoice2TTS(
            model_dir=config.get("TTS_MODEL_DIR", ""),
            voice_ref=config.get("TTS_VOICE_REF", ""),
            voice_ref_text=config.get("TTS_VOICE_REF_TEXT", ""),
        )
    if backend == "dummy":
        return DummyTTS()
    raise ValueError(f"未知 TTS_BACKEND: {backend!r}（可选：dummy / cosyvoice2 / none）")
