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
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

# CosyVoice2 模型：HuggingFace 仓库与默认本地目录
COSYVOICE2_HF_REPO = "FunAudioLLM/CosyVoice2-0.5B"
DEFAULT_MODEL_DIR = "assets/models/CosyVoice2-0.5B"
COSYVOICE2_SAMPLE_RATE = 22050  # CosyVoice2 输出采样率


def _split_sentences(text):
    """按中文句末标点切句（保留标点）。逐句合成播放 = 句粒度打断进度。"""
    return [s for s in re.split(r"(?<=[。！？!?；;])", text) if s.strip()]


def ensure_cosyvoice_model(model_dir=None):
    """CosyVoice2 模型不存在时自动下载（huggingface_hub）。

    先例：faster-whisper 的"离线优先，本地无缓存才在线下载"（listen.py:69）；
    网络不佳时设 HF_ENDPOINT=https://hf-mirror.com 走镜像（README 已有说明）。
    返回模型目录 Path。
    """
    path = Path(model_dir or DEFAULT_MODEL_DIR)
    if path.is_dir() and any(path.iterdir()):
        return path  # 已有模型：离线可用
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("缺少 huggingface_hub，无法自动下载 CosyVoice2 模型（faster-whisper 依赖已自带）")
        return path
    logger.warning("CosyVoice2 模型不存在，开始自动下载（约 2GB）…%s", COSYVOICE2_HF_REPO)
    snapshot_download(repo_id=COSYVOICE2_HF_REPO, local_dir=str(path))
    return path


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
    """CosyVoice2 本地合成（零样本克隆音色，GPU）。

    - 模型：首次 speak 时惰性加载（约 10~20s；缺失自动下载，尊重 HF_ENDPOINT 镜像）
    - 合成：逐句 inference_zero_shot（流式 chunk），sounddevice 播放（22050Hz）
    - 打断：interrupt() 设停止标志 + sd.stop()（线程安全），返回已播句前缀
    - 线程：模型加载/合成/播放都是阻塞操作，speak 整体丢到线程池，不卡事件循环
    """

    def __init__(self, model_dir: str = "", voice_ref: str = "", voice_ref_text: str = ""):
        # 模型不存在自动下载（huggingface_hub，尊重 HF_ENDPOINT 镜像）；同步阻塞，下载时可见进度
        self.model_dir = str(ensure_cosyvoice_model(model_dir))
        self.voice_ref = voice_ref
        self.voice_ref_text = voice_ref_text
        self._model = None
        self._prompt_speech = None
        self._busy = False
        self._played = ""
        self._stop_event = None

    def _load(self):
        """惰性加载模型与参考音频（同步；调用方负责丢线程）。"""
        if self._model is not None:
            return
        if not self.voice_ref or not self.voice_ref_text:
            raise RuntimeError("CosyVoice2 零样本克隆需要配置 TTS_VOICE_REF 与 TTS_VOICE_REF_TEXT")
        import sys

        sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        logger.warning("加载 CosyVoice2 模型（首次约 10~20s）…")
        self._model = CosyVoice2(self.model_dir, load_jit=False, load_trt=False, fp16=True)
        self._prompt_speech = load_wav(self.voice_ref, 16000)
        logger.warning("CosyVoice2 就绪（音色参考：%s）", self.voice_ref)

    async def speak(self, text: str) -> None:
        if not text:
            return
        try:
            await asyncio.to_thread(self._speak_sync, text)
        except Exception as e:
            logger.error("CosyVoice2 合成失败：%s", e)
        finally:
            self._busy = False

    def _speak_sync(self, text: str) -> None:
        """线程内：加载模型 + 逐句合成 + 播放（interrupt 可随时打断）。"""
        self._load()
        self._busy = True
        self._played = ""
        self._stop_event = threading.Event()
        import sounddevice as sd

        for sentence in _split_sentences(text):
            if self._stop_event.is_set():
                break
            for chunk in self._model.inference_zero_shot(sentence, self.voice_ref_text, self._prompt_speech):
                if self._stop_event.is_set():
                    break
                audio = chunk["tts_speech"].cpu().numpy().flatten()
                sd.play(audio, samplerate=COSYVOICE2_SAMPLE_RATE)
                sd.wait()  # interrupt() 会 sd.stop() → wait 提前返回
            self._played += sentence  # 该句播放完成（或被中断时已尽力播放）
        if not self._stop_event.is_set():
            self._played = text  # 全部播完 = 完整文本

    async def interrupt(self) -> str:
        if self._busy and self._stop_event is not None:
            self._stop_event.set()
            try:
                import sounddevice as sd

                sd.stop()  # 线程安全：立即停止当前播放
            except Exception:
                pass
        prefix, self._played = self._played, ""
        return prefix

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
