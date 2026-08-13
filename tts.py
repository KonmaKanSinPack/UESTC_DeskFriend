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
# 每句尾音填充（秒）：sd.play 流停止时设备缓冲尾音可能被丢（突兀截断），
# 填充静音让截断落在静音里。同时是句间自然停顿的底噪
SILICONFLOW_TAIL_PAD = 0.2
# 就绪判定：这些关键文件齐全才算模型完整（下载中断时目录非空但可能缺文件）
COSYVOICE2_REQUIRED_FILES = ("flow.pt", "hift.pt", "CosyVoice-BlankEN/model.safetensors")


def _split_sentences(text):
    """按中文句末标点切句（保留标点）。逐句合成播放 = 句粒度打断进度。

    过滤纯标点句（如「在呢在呢！！」切出第二句「！」）——发给 TTS API 会
    返回错误（真机踩过：Format not recognised），且这类句无语音内容。
    """
    sentences = [s for s in re.split(r"(?<=[。！？!?；;])", text) if s.strip()]
    return [s for s in sentences if re.search(r"[一-鿿A-Za-z0-9]", s)]


def _load_wav_soundfile(wav, target_sr, min_sr=16000):
    """替代 CosyVoice 的 load_wav：soundfile + librosa 实现。

    签名与官方一致（返回单值 speech，形状 (1, N) float32）——注意不是元组，
    曾因返回 (speech, sr) 导致 frontend 拿到 tuple 报 min() TypeError。
    替换原因：torchaudio 2.9+ 移除 soundfile 后端，torchcodec 无 Windows wheel，
    官方 load_wav 在 Windows 必挂。
    """
    import librosa
    import soundfile as sf
    import torch

    speech, sample_rate = sf.read(wav, dtype="float32")
    if sample_rate != target_sr:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sr)
    return torch.from_numpy(speech).unsqueeze(0)


def ensure_cosyvoice_model(model_dir=None):
    """CosyVoice2 模型不存在时自动下载（huggingface_hub）。

    先例：faster-whisper 的"离线优先，本地无缓存才在线下载"（listen.py:69）；
    网络不佳时设 HF_ENDPOINT=https://hf-mirror.com 走镜像（README 已有说明）。
    返回模型目录 Path。
    """
    path = Path(model_dir or DEFAULT_MODEL_DIR)
    # 完整性检查：关键文件齐全才算就绪（下载中断时目录非空但缺文件，必须补下）
    if all((path / f).exists() for f in COSYVOICE2_REQUIRED_FILES):
        return path  # 已有完整模型：离线可用
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
    def playing(self) -> bool:
        """此刻是否正在播放一句音频（sd.play 进行中）。

        与 busy 的区别：busy = 整个 speak 会话进行中（含句间监听窗口 / 合成期）；
        playing 仅在真正 sd.play 一句时为 True，句间窗口 / 合成期为 False。
        耳的污染判定用它区分「录音叠着真实播放」（真回声）vs「窗口期干净录音」。
        """

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

    @property
    def playing(self) -> bool:
        return self._busy  # Dummy 无真实播放，回退 busy


class CosyVoice2TTS(TTS):
    """CosyVoice2 本地合成（零样本克隆音色，GPU）。⚠️ 未完成：保留待完善。

    已知问题（2026-08-11）：本地推理在 Python 3.12 + 新版依赖栈下输出乱码且
    速度不稳定（6.7s~386s/句）；官方环境（3.10 + requirements 全量）验证正常。
    当前主方案为 SiliconFlowTTS（API 托管，1~2s/句稳定）。本实现待完善方向：
    官方环境常驻服务（.venv-cosyvoice）+ vllm 加速，届时可脱离 API 依赖。

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
        self._playing = False  # 精确“此刻正在 sd.play 一句”；句间窗口/合成期为 False（耳污染判定用）
        self._played = ""
        self._speaking_text = ""  # 本次朗读全文（interrupt 防误报用）
        self._stop_event = None
        self.sentence_done_callback = None  # 句间监听窗口钩子（Mouth 注入，非末句播完调用）

    def _load(self):
        """惰性加载模型与参考音频（同步；调用方负责丢线程）。"""
        if self._model is not None:
            return
        if not self.voice_ref or not self.voice_ref_text:
            raise RuntimeError("CosyVoice2 零样本克隆需要配置 TTS_VOICE_REF 与 TTS_VOICE_REF_TEXT")
        import sys

        # CosyVoice 本体 + third_party 子模块（Matcha-TTS 是其组件，非 PyPI 包）
        sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
        sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli import frontend as cosy_frontend
        from cosyvoice.cli.cosyvoice import CosyVoice2

        # Windows 兼容：CosyVoice 的 load_wav 用 torchaudio(backend=soundfile)，
        # 而 torchaudio 2.9+ 移除了该后端（强制 torchcodec，且 torchcodec 无 Windows
        # wheel）——把 frontend 内部的 load_wav 替换为 soundfile+librosa 实现
        cosy_frontend.load_wav = _load_wav_soundfile

        logger.warning("加载 CosyVoice2 模型（首次约 10~20s）…")
        self._model = CosyVoice2(self.model_dir, load_jit=False, load_trt=False, fp16=True)
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
        self._played = ""
        self._speaking_text = text
        self._stop_event = threading.Event()
        self._busy = True  # 置最后：interrupt 见 busy=True 时 _stop_event 必为本次新事件（防 stale event）

        import sounddevice as sd

        sentences = _split_sentences(text)
        if not sentences:
            return  # 纯标点/表情切句后为空：无可播内容，早退
        for i, sentence in enumerate(sentences, 1):
            if self._stop_event.is_set():
                break
            for chunk in self._model.inference_zero_shot(sentence, self.voice_ref_text, self.voice_ref):
                if self._stop_event.is_set():
                    break
                audio = chunk["tts_speech"].cpu().numpy().flatten()
                self._playing = True
                sd.play(audio, samplerate=COSYVOICE2_SAMPLE_RATE)
                sd.wait()  # interrupt() 会 sd.stop() → wait 提前返回
                self._playing = False
            self._played += sentence  # 该句播放完成（或被中断时已尽力播放）
            if i == len(sentences) and not self._stop_event.is_set():
                self._played = text  # 末句播完即定完整文本，收窄「已完成却被误报打断」竞态
            # 句间监听窗口：非末句播完调用（阻塞 = 播放暂停；窗口内被打断 → 循环顶部 break）
            if self.sentence_done_callback is not None and i < len(sentences):
                self.sentence_done_callback()
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
        if prefix == self._speaking_text:  # 完整播完（含竞态窗口），非打断 → 不误报
            return ""
        return prefix

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def played_text(self) -> str:
        return self._played

    @property
    def playing(self) -> bool:
        return self._playing


class SiliconFlowTTS(TTS):
    """SiliconFlow 托管的 CosyVoice2（OpenAI 兼容 /v1/audio/speech）。

    - 每次请求带 references（参考音频 base64 + 文本）实现零样本克隆，免上传流程
    - response_format=wav 直接 sounddevice 播放；逐句请求保持句粒度打断
    - 网络 1~2s/句，质量与官方一致（本地 CosyVoice2 版本栈不稳，此为当前主方案）
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "FunAudioLLM/CosyVoice2-0.5B",
        voice_ref: str = "",
        voice_ref_text: str = "",
        sample_rate: int = 32000,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice_ref = voice_ref
        self.voice_ref_text = voice_ref_text
        self.sample_rate = sample_rate
        self._ref_b64 = None  # 参考音频 base64（读一次缓存）
        self._busy = False
        self._playing = False  # 精确“此刻正在 sd.play 一句”；句间窗口/合成期为 False（耳污染判定用）
        self._played = ""
        self._speaking_text = ""  # 本次朗读全文（interrupt 防误报用）
        self._stop_event = None
        self.sentence_done_callback = None  # 句间监听窗口钩子（Mouth 注入，非末句播完调用）

    def _ref_audio_b64(self):
        """参考音频 base64（data URI）；读一次缓存，避免每次请求读盘。"""
        if self._ref_b64 is None:
            import base64

            with open(self.voice_ref, "rb") as f:
                self._ref_b64 = f"data:audio/wav;base64,{base64.b64encode(f.read()).decode()}"
        return self._ref_b64

    async def speak(self, text: str) -> None:
        if not text:
            return
        try:
            await asyncio.to_thread(self._speak_sync, text)
        except Exception as e:
            logger.error("SiliconFlow TTS 合成失败：%s", e)
        finally:
            self._busy = False

    def _speak_sync(self, text: str) -> None:
        """线程内：并行请求逐句合成，按序连续播放（interrupt 可随时打断）。

        句间零网络静默（曾每句串行 HTTP，句间 1.5s 死寂 = 用户感知的"突兀截断"）：
        所有句子并发请求，播放时只等当前句（fut.result()），后续句的请求在
        上一句播放期间已完成。打断语义不变：循环顶部查 stop_event，break 后
        线程池 shutdown(wait=False) 不等在途请求，不阻塞打断返回。
        """
        import threading as _th
        import time
        from concurrent.futures import ThreadPoolExecutor
        from io import BytesIO

        import httpx
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        if not self.voice_ref or not self.voice_ref_text:
            raise RuntimeError("SiliconFlow 克隆音色需要配置 TTS_VOICE_REF 与 TTS_VOICE_REF_TEXT")
        self._played = ""
        self._speaking_text = text
        self._stop_event = threading.Event()
        self._busy = True  # 置最后：interrupt 见 busy=True 时 _stop_event 必为本次新事件（防 stale event）
        headers = {"Authorization": f"Bearer {self.api_key}"}
        sentences = _split_sentences(text)
        if not sentences:
            return  # 纯标点/表情切句后为空：无可播内容，早退（防 ThreadPoolExecutor(max_workers=0) 崩）
        print(f"[TTS] 开始朗读（{len(sentences)} 句，线程 {_th.current_thread().name}）")

        def fetch(sentence):
            """单句合成请求（线程池内并发执行）。"""
            payload = {
                "model": self.model,
                "input": sentence,
                "references": [{"audio": self._ref_audio_b64(), "text": self.voice_ref_text}],
                "response_format": "wav",
                "sample_rate": self.sample_rate,
            }
            resp = httpx.post(f"{self.base_url}/audio/speech", headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            try:
                audio, sr = sf.read(BytesIO(resp.content))
            except Exception as e:
                # API 返回非 wav（如错误 JSON）时带上响应片段，便于定位
                raise RuntimeError(f"wav 解析失败：{e}（响应片段：{resp.content[:200]!r}）") from e
            return audio, sr, sentence

        pad = int(SILICONFLOW_TAIL_PAD * self.sample_rate)
        pool = ThreadPoolExecutor(max_workers=min(len(sentences), 4))
        try:
            futures = [pool.submit(fetch, s) for s in sentences]
            for i, fut in enumerate(futures, 1):
                if self._stop_event.is_set():
                    break
                t0 = time.monotonic()
                try:
                    audio, sr, sentence = fut.result()  # 只等当前句；其余句并行请求中
                except Exception as e:
                    print(f"[TTS] 句{i} 合成失败：{sentences[i - 1]!r} → {e}")  # 单句失败不拖垮整段
                    continue
                t1 = time.monotonic()
                # 尾音填充：流停止时设备缓冲尾音可能被丢，静音垫底防"突兀截断"
                audio = np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])
                wav_len = len(audio) / sr
                # 尾部 RMS 诊断：wav 结尾是否自然衰减（尾 RMS 高 = 合成/服务端截断；
                # 尾 RMS 低 = wav 干净，截断发生在播放层流停止）
                seg = int(0.05 * sr)
                tail_rms = float(np.sqrt(np.mean(np.asarray(audio[-seg:]) ** 2))) if len(audio) else 0.0
                head_rms = float(np.sqrt(np.mean(np.asarray(audio[:seg]) ** 2))) if len(audio) else 0.0
                self._playing = True
                sd.play(audio, sr)
                t2 = time.monotonic()
                sd.wait()  # interrupt() 会 sd.stop() → wait 提前返回
                self._playing = False
                t3 = time.monotonic()
                flag = ""
                if t3 - t2 < wav_len - 0.2:
                    flag = " <- 播放被提前终止"
                elif t3 - t2 > wav_len + 0.3:
                    flag = " <- 播放超时(underrun?)"
                print(
                    f"[TTS] 句{i}/{len(sentences)} 等待{t1 - t0:.2f}s wav{wav_len:.2f}s "
                    f"播放{t3 - t2:.2f}s 头RMS{head_rms:.3f} 尾RMS{tail_rms:.3f}{flag}"
                )
                self._played += sentence  # 该句播放完成（或被中断）
                if i == len(sentences) and not self._stop_event.is_set():
                    self._played = text  # 末句播完即定完整文本，收窄「已完成却被误报打断」竞态
                # 句间监听窗口：非末句播完调用（阻塞 = 播放暂停；窗口内被打断 → 循环顶部 break）
                if self.sentence_done_callback is not None and i < len(sentences):
                    self.sentence_done_callback()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # 不阻塞打断返回
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
        if prefix == self._speaking_text:  # 完整播完（含竞态窗口），非打断 → 不误报
            print(f"[TTS] interrupt 未生效（已完成播放，{len(prefix)} 字符）")
            return ""
        if prefix:
            print(f"[TTS] 朗读被打断（已播 {len(prefix)} 字符）")
        return prefix

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def played_text(self) -> str:
        return self._played

    @property
    def playing(self) -> bool:
        return self._playing


def create_tts(config) -> TTS:
    """按配置装配 TTS 后端（照 backends 工厂模式）。"""
    backend = config.get("TTS_BACKEND", "dummy")
    if backend == "none":
        return DummyTTS()  # none 与 dummy 等价：占位链路可用，不发声
    if backend == "siliconflow":
        return SiliconFlowTTS(
            api_key=config.get("TTS_API_KEY", ""),
            base_url=config.get("TTS_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=config.get("TTS_MODEL", "FunAudioLLM/CosyVoice2-0.5B"),
            voice_ref=config.get("TTS_VOICE_REF", ""),
            voice_ref_text=config.get("TTS_VOICE_REF_TEXT", ""),
            sample_rate=config.get("TTS_SAMPLE_RATE", 32000),
        )
    if backend == "cosyvoice2":
        return CosyVoice2TTS(
            model_dir=config.get("TTS_MODEL_DIR", ""),
            voice_ref=config.get("TTS_VOICE_REF", ""),
            voice_ref_text=config.get("TTS_VOICE_REF_TEXT", ""),
        )
    if backend == "dummy":
        return DummyTTS()
    raise ValueError(f"未知 TTS_BACKEND: {backend!r}（可选：siliconflow / cosyvoice2 / dummy / none）")
