import threading
from collections import deque
from pathlib import Path

import ctranslate2
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt5.QtCore import QObject, pyqtSignal

# 句间监听窗口：打断需活动度确认——最近 10 块（320ms）中 ≥8 块活跃才算用户说话。
# 真机实测：播放"呜哇！"后余响尖峰仅 160ms（5 块）连续，不足以触发确认；
# 真说话（两音节以上）在任意 320ms 窗口内活跃块必 ≥8。
WINDOW_ACTIVITY_HISTORY = 10  # 活动度窗口（块数，32ms/块）
WINDOW_INTERRUPT_ACTIVE = 8  # 打断阈值：窗口内活跃块数


def should_drop_echo(speaking: bool, window_open: bool) -> bool:
    """门控规则（句间监听窗口方案）：VAD 触发时是否视为回声丢弃（纯逻辑便于单测）。

    - 嘴未发声 → 不丢（正常监听）
    - 嘴发声且监听窗口未开（句子播放中 / 句间余响静默期）→ 丢：此刻拾到的
      必是扬声器回声（播放中或上一句余响），若当真会触发 interrupt 打断自己
    - 嘴发声且监听窗口开着 → 不丢：窗口期 = 只此期间拾音，再经活动度确认
    """
    return speaking and not window_open


def window_interrupt_confirmed(activity, active_threshold=WINDOW_INTERRUPT_ACTIVE) -> bool:
    """窗口期打断确认：最近 WINDOW_ACTIVITY_HISTORY 块中活跃 ≥ 阈值才算用户说话。

    余响是衰减尖峰（真机实测 ~160ms = 5 块），真说话持续更久——用活动度
    而非单块触发，防余响尖峰自打断。
    """
    return sum(activity) >= active_threshold


def segment_contaminated(playing_flags) -> bool:
    """录音段内任意时刻叠着 TTS 播放 → 本段必混回声，转写结果不可信。

    注意用「正在播放」（mouth.busy）而非「会话中」（speaking）判断——句间
    监听窗口内 speaking 恒为 True，若用 speaking 会把窗口期录音误判为污染
    而丢弃（真机踩过：打断成功但用户的话永远到不了转写）。
    """
    return any(playing_flags)


class SileroVadOnnx:
    """直接调用 silero_vad.onnx 打分，不依赖 torch。

    模型文件内置在 assets/ 下（来自 silero-vad 官方包，MIT 协议），
    运行时无需联网下载。每次调用维护隐状态 state，分段录音前应 reset()。
    """

    def __init__(self, sample_rate=16000):
        onnx_path = Path(__file__).parent / "assets" / "silero_vad.onnx"
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.sr = np.array(sample_rate, dtype=np.int64)

        # ---- 从模型元数据推导 state 的 shape ----
        # 模型输出 [2, None, 128]；None 是动态 batch 维度，推理时固定为 1
        state_meta = next(i for i in self.session.get_inputs() if i.name == "state")
        self._state_shape = tuple(d if isinstance(d, int) else 1 for d in state_meta.shape)

        # v5 模型要求在每块音频前拼接上一块的末尾作为上下文（16kHz 为 64 采样点）
        self.context_size = 64 if sample_rate == 16000 else 32
        self.reset()

    def reset(self):
        self.state = np.zeros(self._state_shape, dtype=np.float32)
        self.context = np.zeros((1, self.context_size), dtype=np.float32)

    def __call__(self, chunk_f32):
        """chunk_f32: (512,) 的 float32 音频块，返回语音概率。"""
        x = np.concatenate([self.context, chunk_f32[None, :]], axis=1)
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[:, -self.context_size :]
        return float(out[0, 0])


class Listen(QObject):
    text_signal = pyqtSignal(str)  # 只是信号通道，不是消息缓存。
    interrupt_requested = pyqtSignal()  # 句间监听窗口内 VAD 触发 = 用户说话 → 主控立即打断

    def __init__(self, history_length=5):
        super().__init__()
        self.listen_history = deque(maxlen=history_length)
        # 回声门控：嘴器官引用（ui 装配）。朗读期间麦克风拾到的语音视为回声丢弃，
        # 门控状态由 Mouth 内部管理（speaking 会话 + window_open 句间监听窗口），
        # 耳只读不写
        self.mouth = None
        # 窗口期打断活动度：最近 N 块 VAD 结果（True=活跃）；确认后才打断
        self._window_activity = deque(maxlen=WINDOW_ACTIVITY_HISTORY)
        self._interrupt_sent = False  # 本窗口是否已发打断（防重复触发）

        # 音频配置参数 (VAD 要求的标准格式)
        self.SAMPLE_RATE = 16000  # 采样率：16kHz
        self.CHUNK = 512  # 每次读取的音频块大小

        # 开启麦克风数据流
        self.stream = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,  # 16000 Hz
            channels=1,  # 单声道
            dtype="int16",  # 16位深度
            blocksize=self.CHUNK,  # 每次读取 512 帧
        )
        self.stream.start()

        # 加载 Silero VAD（ONNX 本地模型，无需 torch、无需联网下载）
        self.vad = SileroVadOnnx(self.SAMPLE_RATE)

        # 加载whisper：有 CUDA 用 GPU，否则回退 CPU int8
        if ctranslate2.get_cuda_device_count() > 0:
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
        print(f"Whisper 推理设备：{device} ({compute_type})")
        try:
            # 优先离线加载本地缓存：在线模式即使模型已缓存也会先连 HF 校验，
            # 网络不通时会卡死在 TCP 连接上
            self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type, local_files_only=True)
        except Exception:
            print("本地未找到 Whisper 模型缓存，转为在线下载...")
            self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type)

        self.start_threading()

    @property
    def _speaking(self):
        """嘴是否在发声（含余响尾巴）；未装配（None）视为不发声。"""
        return self.mouth.speaking if self.mouth is not None else False

    @property
    def _window_open(self):
        """嘴的句间监听窗口是否开着（窗口内耳才拾音）；未装配（None）视为关。"""
        return self.mouth.window_open if self.mouth is not None else False

    def _window_interrupt_msg(self) -> str:
        """窗口期打断日志（活动度摘要）。"""
        return f"监听窗口内检测到用户说话，打断朗读（{sum(self._window_activity)}/{WINDOW_ACTIVITY_HISTORY} 块活跃）"

    def start_threading(self):
        # daemon=True意思是：这个线程是个守护线程，主线程结束了它也会跟着结束，不会阻碍程序退出。
        listen_thread = threading.Thread(target=self.during_listening, daemon=True)
        listen_thread.start()

    def get_voice_text(self, text):
        # 广播机制：只要发出信号，任何监听这个信号的对象都能收到并处理这个文本消息
        self.text_signal.emit(text)

    def during_listening(self):
        silence_timeout = 0
        # 每块 512 帧 @16kHz ≈ 32ms
        MAX_SILENCE = 45  # 连续静音约 1.5 秒视为说完
        MAX_CHUNKS = 470  # 最长录音约 15 秒，防止缓冲无限增长
        echo_ignored = False  # 正在忽略回声（防每 32ms 重复打印）
        while True:
            voice_buffer = []
            raw_bytes, _overflowed = self.stream.read(self.CHUNK)
            audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # 模型打分
            score = self.vad(audio_f32)
            # 句间监听窗口：活动度累积（每块都记，非窗口期复位）
            if self._speaking and self._window_open:
                self._window_activity.append(score >= 0.5)
                if window_interrupt_confirmed(self._window_activity) and not self._interrupt_sent:
                    # 最近 320ms 持续活跃 = 真用户说话（余响是单发尖峰，够不着阈值）
                    self._interrupt_sent = True
                    self.interrupt_requested.emit()
                    print(self._window_interrupt_msg())
            else:
                if self._window_activity or self._interrupt_sent:
                    self._window_activity.clear()
                    self._interrupt_sent = False
            if score >= 0.5:
                # 句间监听窗口方案：句子播放中/余响静默期（speaking 且窗口未开）→
                # 拾到的必是扬声器回声，忽略（不 reset VAD、不录音，继续读流保持同步）；
                # 窗口期 → 录音（打断由活动度确认触发）
                if should_drop_echo(self._speaking, self._window_open):
                    if not echo_ignored:
                        print("检测到 TTS 播放回声，忽略")
                        echo_ignored = True
                    continue
                echo_ignored = False
                print("检测到声音了，开始录音...")
                self.vad.reset()  # 每段录音前重置 VAD 状态
                voice_buffer.append(audio_f32)
                playing_flags = []  # 逐块记录：录音段是否叠着 TTS 播放
                while silence_timeout < MAX_SILENCE and len(voice_buffer) < MAX_CHUNKS:
                    raw_bytes, _overflowed = self.stream.read(self.CHUNK)
                    audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # 模型打分
                    score = self.vad(audio_f32)

                    # 窗口期活动度继续累积（打断确认可能发生在录音中）
                    if self._speaking and self._window_open:
                        self._window_activity.append(score >= 0.5)
                        if window_interrupt_confirmed(self._window_activity) and not self._interrupt_sent:
                            self._interrupt_sent = True
                            self.interrupt_requested.emit()
                            print(self._window_interrupt_msg())

                    playing_flags.append(self.mouth.busy if self.mouth is not None else False)

                    if score < 0.5:
                        silence_timeout += 1

                    else:
                        silence_timeout = 0

                    voice_buffer.append(audio_f32)

                silence_timeout = 0
                if segment_contaminated(playing_flags):
                    # 录音中桃桃开播下一句：本段混入播放声，不转写不 emit
                    print("录音段叠着 TTS 播放，丢弃（疑似回声）")
                    continue
                print("录音结束，正在转写...")
                complete_audio_np = np.concatenate(voice_buffer) if voice_buffer else np.zeros(0, dtype=np.float32)
                # 交给 Whisper 转写（float32 原始信号；窗口期录音已无播放混叠）
                segments, info = self.whisper_model.transcribe(complete_audio_np, beam_size=5, language="zh")
                transed_text = "".join([segment.text for segment in segments])
                if transed_text.strip():
                    self.get_voice_text(transed_text)
