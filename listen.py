import threading
from collections import deque
from pathlib import Path

import ctranslate2
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt5.QtCore import QObject, pyqtSignal


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

    def __init__(self, history_length=5):
        super().__init__()
        self.listen_history = deque(maxlen=history_length)

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
        while True:
            voice_buffer = []
            raw_bytes, _overflowed = self.stream.read(self.CHUNK)
            audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # 模型打分
            score = self.vad(audio_f32)
            if score >= 0.5:
                print("检测到声音了，开始录音...")
                self.vad.reset()  # 每段录音前重置 VAD 状态
                voice_buffer.append(raw_bytes)
                while silence_timeout < MAX_SILENCE and len(voice_buffer) < MAX_CHUNKS:
                    raw_bytes, _overflowed = self.stream.read(self.CHUNK)
                    audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # 模型打分
                    score = self.vad(audio_f32)

                    if score < 0.5:
                        silence_timeout += 1

                    else:
                        silence_timeout = 0

                    voice_buffer.append(raw_bytes)

                print("录音结束，正在转写...")
                complete_audio_bytes = b"".join(voice_buffer)
                complete_audio_np = np.frombuffer(complete_audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                # 把 complete_audio_bytes 交给 Whisper 转写
                segments, info = self.whisper_model.transcribe(complete_audio_np, beam_size=5, language="zh")
                transed_text = "".join([segment.text for segment in segments])
                if transed_text.strip():
                    self.get_voice_text(transed_text)
                silence_timeout = 0
