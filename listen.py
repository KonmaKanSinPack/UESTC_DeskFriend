import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from PyQt5.QtCore import QObject, pyqtSignal

from asr import create_asr

# 句间监听窗口：打断需活动度确认——最近 10 块（320ms）中 ≥8 块活跃才算用户说话。
# 真机实测：播放"呜哇！"后余响尖峰仅 160ms（5 块）连续，不足以触发确认；
# 真说话（两音节以上）在任意 320ms 窗口内活跃块必 ≥8。
WINDOW_ACTIVITY_HISTORY = 10  # 活动度窗口（块数，32ms/块）
WINDOW_INTERRUPT_ACTIVE = 8  # 打断阈值：窗口内活跃块数
# 打断冷却（秒）：打断后短期内抑制再次打断（防 cancel-restart 循环，照
# FutureAGI barge-in 指南：打断后再打断的最小间隔 ~350ms）
INTERRUPT_COOLDOWN = 0.35


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
    """录音段内任意时刻叠着 TTS 真实播放 → 本段必混回声，转写结果不可信。

    用「此刻正在播一句」（mouth.playing）判断，而非「会话中」（speaking）或
    「speak 进行中」（busy）——句间监听窗口内 speaking 与 busy 都恒为 True，
    用它们会把窗口期干净录音整段误判污染丢弃（真机踩过：打断成功但用户的话
    永远到不了转写）。playing 只在 sd.play 一句时为 True、句间/合成期为 False，
    故只丢真正叠进下一句播放的录音段。
    """
    return any(playing_flags)


# ---- 音频三小修（2026-09-03，引擎无关的采集侧增强；纯函数/纯类便于单测）----

# pre-roll 环形缓冲块数：7 块 ≈ 224ms @512帧/16kHz。VAD 触发判定需要能量爬坡，
# 触发那刻话音往往已说了一两百毫秒——没有 pre-roll 首字/半字被切，识别白白丢分
PRE_ROLL_CHUNKS = 7
# 裁尾保留块数：留 ~96ms 自然停顿感；其余尾部静音裁掉（静音段是转写幻觉温床）
TRIM_KEEP_CHUNKS = 3


class PreRollBuffer:
    """触发前音频的环形缓冲：VAD 触发那刻之前的话音（首字前导）不丢失。

    只收非回声块（嘴在播且监听窗口未开时不收）——防回声期块混进下一段录音
    开头，绕过 segment_contaminated 的播放标志判定。
    drain() 取走并清空：触发即消费，缓冲不跨段残留。
    """

    def __init__(self, max_chunks=PRE_ROLL_CHUNKS):
        self._buf = deque(maxlen=max_chunks)

    def append(self, chunk, playing=False):
        self._buf.append((chunk, playing))

    def drain(self):
        chunks = [c for c, _ in self._buf]
        playing_flags = [p for _, p in self._buf]
        self._buf.clear()
        return chunks, playing_flags


def trim_trailing_silence(chunks, scores, threshold=0.5, keep=TRIM_KEEP_CHUNKS):
    """裁掉录音尾部的连续静音块，保留 keep 块自然停顿。

    录音以连续静音 MAX_SILENCE(1.5s) 收尾才判"说完"，这 1.5s 静音如果一起送
    转写，是幻觉的温床（模型对无内容段容易编造"谢谢观看"类文本）。按录音期
    记录的逐块 VAD 分数，把尾部低于阈值的块全部裁掉、只留 keep 块——既消灭
    幻觉温床，又保留自然停顿的听感特征。scores 与 chunks 一一对应。
    """
    end = len(chunks)
    while end > 0 and scores[end - 1] < threshold:
        end -= 1
    end = min(len(chunks), end + keep)
    return chunks[:end]


def boost_if_quiet(audio, peak_threshold=0.25, target=0.5):
    """小音量增益：整段峰值低于阈值才等比放大到目标（保守规则）。

    原理：远场麦克风/系统音量低时录到的信号峰值可能只有 0.1x，转写特征
    提取偏弱。只在"确实很小声"时放大——正常音量不动，避免把底噪一起抬起来。
    """
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if 0.0 < peak < peak_threshold:
        return audio * (target / peak)
    return audio


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

    def __init__(self, history_length=5, config=None, asr_engine=None):
        super().__init__()
        self.listen_history = deque(maxlen=history_length)
        # ASR 引擎可注入（测试用）；生产路径从 config.toml 经工厂装配
        # （ASR_BACKEND 选 SenseVoice/Whisper，缺模型自动下载/回退，见 asr.py）
        self.asr = asr_engine or create_asr(config)
        # pre-roll 环形缓冲：VAD 触发前的音频块，触发时前置进录音段保首字
        self.pre_roll = PreRollBuffer()
        # 回声门控：嘴器官引用（ui 装配）。朗读期间麦克风拾到的语音视为回声丢弃，
        # 门控状态由 Mouth 内部管理（speaking 会话 + window_open 句间监听窗口），
        # 耳只读不写
        self.mouth = None
        # 窗口期打断活动度：最近 N 块 VAD 结果（True=活跃）；确认后才打断
        self._window_activity = deque(maxlen=WINDOW_ACTIVITY_HISTORY)
        self._interrupt_sent = False  # 本窗口是否已发打断（防重复触发）
        self._prev_window_open = False  # 窗口开合沿检测（开窗时重置 VAD 状态）
        self._interrupt_cooldown_until = 0.0  # 打断冷却截止（monotonic）
        self._prev_speaking = False  # 朗读会话开合沿：新朗读段清打断闩/冷却（防跨回复残留）

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

    def _window_tick(self, score: float) -> None:
        """句间监听窗口每块处理：新段清闩 → 开窗重置 VAD → 活动度累积 → 冷却 → 打断。

        - 朗读会话沿（speaking False→True）：清打断闩/冷却/活动度，避免上一段的
          冷却或 latch 残留抑制本段首个窗口（连续回复时第二段首窗打不断的根因）
        - 开窗沿：vad.reset() 丢弃播放期被回声污染的状态（voice-echo 建议，
          否则窗口期打分从脏状态开始，余响更易误判）
        - 活动度：最近 10 块 ≥8 活跃才打断；打断后 INTERRUPT_COOLDOWN 内
          不再打断（防 cancel-restart 循环，FutureAGI 指南）
        """
        speaking = self._speaking
        if speaking and not self._prev_speaking:
            # 新一段朗读开始：清打断闩与冷却，避免上一段的冷却/latch 抑制本段首个窗口
            self._interrupt_sent = False
            self._interrupt_cooldown_until = 0.0
            self._window_activity.clear()
        self._prev_speaking = speaking
        window_open = self._window_open
        if window_open != self._prev_window_open:
            if window_open:
                self.vad.reset()  # 开窗：干净状态打分
            self._prev_window_open = window_open
        if speaking and window_open:
            self._window_activity.append(score >= 0.5)
            now = time.monotonic()
            if (
                window_interrupt_confirmed(self._window_activity)
                and not self._interrupt_sent
                and now >= self._interrupt_cooldown_until
            ):
                self._interrupt_sent = True
                self._interrupt_cooldown_until = now + INTERRUPT_COOLDOWN
                self.interrupt_requested.emit()
                print(self._window_interrupt_msg())
        else:
            if self._window_activity or self._interrupt_sent:
                self._window_activity.clear()
                self._interrupt_sent = False

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
        # 连续静音 1.0s 判"说完"（2026-09-03 流畅性轮 1.5s→1.0s，行业典型 0.7~1.0s）。
        # 早年 640ms 曾把说话停顿截断——现在安全得多：截出来的后半句会作为新消息入队，
        # 被 consumer 与前半段 "\n" 合并成一条送脑，不再丢内容
        MAX_SILENCE = 30
        MAX_CHUNKS = 470  # 最长录音约 15 秒，防止缓冲无限增长
        echo_ignored = False  # 正在忽略回声（防每 32ms 重复打印）
        while True:
            raw_bytes, _overflowed = self.stream.read(self.CHUNK)
            audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            # 模型打分
            score = self.vad(audio_f32)
            # 句间监听窗口：开窗重置 VAD / 活动度累积 / 冷却 / 打断
            self._window_tick(score)
            # 回声期的块不进 pre-roll（防回声混进下一段开头绕过污染判定）
            is_echo_now = should_drop_echo(self._speaking, self._window_open)
            if not is_echo_now:
                self.pre_roll.append(audio_f32, self.mouth.playing if self.mouth is not None else False)
            if score >= 0.5:
                # 句间监听窗口方案：句子播放中/余响静默期（speaking 且窗口未开）→
                # 拾到的必是扬声器回声，忽略（不 reset VAD、不录音，继续读流保持同步）；
                # 窗口期 → 录音（打断由活动度确认触发）
                if is_echo_now:
                    if not echo_ignored:
                        print("检测到 TTS 播放回声，忽略")
                        echo_ignored = True
                    continue
                echo_ignored = False
                print("检测到声音了，开始录音...")
                self.vad.reset()  # 每段录音前重置 VAD 状态
                # pre-roll 前置（保首字）：触发前 ~224ms 的干净块接在本段开头；
                # 这些块在 VAD 阈值下（分数记 0.0），只参与识别不参与静音判定
                pre_chunks, pre_playing = self.pre_roll.drain()
                voice_buffer = pre_chunks + [audio_f32]
                chunk_scores = [0.0] * len(pre_chunks) + [score]
                playing_flags = pre_playing + [self.mouth.playing if self.mouth is not None else False]
                while silence_timeout < MAX_SILENCE and len(voice_buffer) < MAX_CHUNKS:
                    raw_bytes, _overflowed = self.stream.read(self.CHUNK)
                    audio_f32 = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                    # 模型打分
                    score = self.vad(audio_f32)

                    # 窗口期活动度继续累积（打断确认可能发生在录音中）
                    self._window_tick(score)

                    playing = self.mouth.playing if self.mouth is not None else False
                    playing_flags.append(playing)
                    chunk_scores.append(score)

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
                # 三小修后两步：裁掉尾部静音（幻觉温床）→ 小音量增益；
                # （第一步 pre-roll 已在触发时前置）
                voice_buffer = trim_trailing_silence(voice_buffer, chunk_scores)
                if not voice_buffer:
                    continue
                complete_audio_np = boost_if_quiet(np.concatenate(voice_buffer))
                # 交给 ASR 引擎转写（float32 原始信号；窗口期录音已无播放混叠）
                transed_text = self.asr.transcribe(complete_audio_np, self.SAMPLE_RATE)
                if transed_text.strip():
                    self.get_voice_text(transed_text)
