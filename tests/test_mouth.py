"""Mouth 发声器官测试：注入 FakeTTS 验证门控生命周期（不真发声）。

门控语义：speak 置位 speaking 覆盖朗读+余响尾巴，try/finally 保证打断/异常
路径也释放；finished 信号在朗读结束（含被打断）时广播一次。
"""

import asyncio

import pytest

from mouth import TTS_ECHO_TAIL, Mouth


class FakeTTS:
    """假发声后端：可配置播放时长与是否抛错。"""

    def __init__(self, duration=0.02, raise_error=False):
        self.duration = duration
        self.raise_error = raise_error
        self._busy = False
        self._played = ""
        self.interrupt_calls = 0

    async def speak(self, text):
        if self.raise_error:
            raise RuntimeError("后端挂了")
        self._busy = True
        self._played = text
        await asyncio.sleep(self.duration)
        self._busy = False

    async def interrupt(self):
        self.interrupt_calls += 1
        return "第一句。"  # 固定打断位置（发声状态由 busy/played_text 承载）

    @property
    def busy(self):
        return self._busy

    @property
    def played_text(self):
        return self._played


@pytest.fixture
def mouth():
    m = Mouth(tts=FakeTTS())
    yield m


class TestSpeak:
    def test_gate_covers_playback_and_tail(self, mouth):
        """speaking 置位覆盖朗读 + 余响尾巴，之后释放。"""
        events = []

        async def run():
            task = asyncio.create_task(mouth.speak("你好呀"))
            await asyncio.sleep(0.01)  # 朗读中
            events.append(("mid", mouth.speaking))
            await asyncio.sleep(TTS_ECHO_TAIL + 0.05)  # 尾巴已过
            await task
            events.append(("after", mouth.speaking))

        asyncio.run(run())
        assert events == [("mid", True), ("after", False)]

    def test_finished_emitted_once(self, mouth):
        emitted = []

        mouth.finished.connect(lambda: emitted.append(True))
        asyncio.run(mouth.speak("你好呀"))
        assert emitted == [True]

    def test_empty_text_noop(self, mouth):
        """空文本：不置位门控、不发信号。"""
        emitted = []
        mouth.finished.connect(lambda: emitted.append(True))
        asyncio.run(mouth.speak(""))
        assert mouth.speaking is False
        assert emitted == []

    def test_gate_released_on_backend_error(self):
        """后端异常：finally 仍释放门控并广播结束（防门控卡死）。"""
        m = Mouth(tts=FakeTTS(raise_error=True))
        emitted = []
        m.finished.connect(lambda: emitted.append(True))
        with pytest.raises(RuntimeError):
            asyncio.run(m.speak("你好呀"))
        assert m.speaking is False
        assert emitted == [True]

    def test_speak_delegates_to_backend(self, mouth):
        assert asyncio.run(mouth.speak("你好呀")) is None  # 正常返回，不抛错


class TestInterrupt:
    def test_interrupt_returns_prefix(self, mouth):
        prefix = asyncio.run(mouth.interrupt())
        assert prefix == "第一句。"  # 转发后端返回的打断位置

    def test_busy_and_played_text_delegate(self, mouth):
        assert mouth.busy is mouth.tts.busy
        assert mouth.played_text == mouth.tts.played_text


class TestReferenceBuffer:
    """AEC 参考信号：播放音频 tee 成 16k 块缓冲，耳按延迟窗口 drain。"""

    def test_tee_resamples_and_chunks(self, mouth):
        """32k 正弦 0.1s → 16k 块（512/块，尾块补零对齐）。"""
        import numpy as np

        sr = 32000
        audio = np.sin(np.arange(int(0.1 * sr)) * 0.1).astype(np.float32)
        mouth._tee_reference(audio, sr)
        blocks = [b for _, b in mouth._ref_buf]
        assert len(blocks) == 4  # 1600 样本 → 3 整块 + 1 补零尾块
        assert sum(len(b) for b in blocks) == 4 * 512  # 补零后整块对齐
        assert all(len(b) == 512 for b in blocks)  # 尾块补零对齐麦克风块长

    def test_tee_16k_passthrough(self, mouth):
        import numpy as np

        audio = np.ones(512, dtype=np.float32)
        mouth._tee_reference(audio, 16000)
        assert len(mouth._ref_buf) == 1
        assert mouth._ref_buf[0][1][0] == 1.0

    def test_tee_uses_play_time_stamp(self, mouth):
        """t_play 由后端在 sd.play 前记录，块时间戳用它（对齐比 tee 时刻准）。"""
        import time

        import numpy as np

        t0 = time.monotonic()
        mouth._tee_reference(np.ones(512, dtype=np.float32), 16000, t_play=t0)
        assert mouth._ref_buf[0][0] == t0

    def test_drain_returns_block_played_delay_ago(self, mouth):
        """只取播放时刻 ≤ now-delay 的块（延迟窗口对齐），未来块不取。"""
        import time

        import numpy as np

        now = time.monotonic()
        mouth._ref_buf.append((now - 0.2, np.ones(512, dtype=np.float32)))
        mouth._ref_buf.append((now, np.full(512, 2.0, dtype=np.float32)))
        ref = mouth.drain_reference(now, delay=0.15)
        assert ref is not None and ref[0] == 1.0

    def test_drain_none_when_empty(self, mouth):
        assert mouth.drain_reference() is None

    def test_speak_clears_reference_buffer(self, mouth):
        """朗读结束清参考缓冲（防陈旧参考被当成回声去消）。"""
        import numpy as np

        mouth._ref_buf.append((0.0, np.zeros(512, dtype=np.float32)))
        asyncio.run(mouth.speak("你好呀"))
        assert len(mouth._ref_buf) == 0
        assert mouth.speak_started_at > 0

    def test_wires_ref_callback_to_backend(self):
        """后端有 ref_callback 属性时，Mouth 注入 tee 钩子。"""

        class BackendWithRef:
            def __init__(self):
                self.ref_callback = None

            async def speak(self, text):
                pass

            async def interrupt(self):
                return ""

            @property
            def busy(self):
                return False

            @property
            def played_text(self):
                return ""

        b = BackendWithRef()
        m = Mouth(tts=b)
        assert b.ref_callback == m._tee_reference  # 绑定方法用 ==（每次访问是新对象）
