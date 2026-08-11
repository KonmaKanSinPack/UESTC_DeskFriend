"""Mouth 发声器官测试：注入 FakeTTS 验证门控生命周期（不真发声）。

门控语义：speak 置位 speaking 覆盖朗读+余响尾巴，try/finally 保证打断/异常
路径也释放；finished 信号在朗读结束（含被打断）时广播一次。
"""

import asyncio
import time

import pytest

from mouth import SENTENCE_GUARD, SENTENCE_WINDOW, TTS_ECHO_TAIL, Mouth, echo_like


class FakeTTS:
    """假发声后端：可配置播放时长与是否抛错。"""

    def __init__(self, duration=0.02, raise_error=False):
        self.duration = duration
        self.raise_error = raise_error
        self._busy = False
        self._played = ""
        self.interrupt_calls = 0
        self.speak_calls = []

    async def speak(self, text):
        if self.raise_error:
            raise RuntimeError("后端挂了")
        self.speak_calls.append(text)
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


class TestSpeakSerialization:
    def test_concurrent_speaks_serialize(self):
        """连续回复（双击/快速消息）按序朗读，不双流叠播。"""
        fake = FakeTTS(duration=0.05)
        m = Mouth(tts=fake)

        async def run():
            await asyncio.gather(m.speak("第一段"), m.speak("第二段"))
            return fake.speak_calls

        calls = asyncio.run(run())
        assert calls == ["第一段", "第二段"]  # 按序
        assert m.speaking is False  # 门控最终释放


class TestEchoLike:
    def test_similar_to_played_is_echo(self):
        """余响转写（简繁混写）与刚朗读内容相似 → 判回声。"""
        assert echo_like("桃桃沒有回應", "（桃桃没有回应…）") is True

    def test_identical_is_echo(self):
        assert echo_like("今天天气不错", "今天天气不错。") is True

    def test_unrelated_not_echo(self):
        assert echo_like("我们下课去吃饭吧", "今天天气不错，我们一起去公园散步吧。") is False

    def test_empty_played_not_echo(self):
        assert echo_like("你好", "") is False


class TestSentenceWindow:
    """句间监听窗口：guard 静默 → 窗口开（耳拾音）→ 关；后端钩子注入。"""

    def test_gap_flips_window_open_state(self, mouth):
        """_sentence_gap：窗口在 guard 后打开、在窗口期结束后关闭。"""
        mouth.window_open = False
        t0 = time.monotonic()
        mouth._sentence_gap()
        elapsed = time.monotonic() - t0
        # Windows time.sleep 粒度可能略短（实测差 ~13ms），留 50ms 容差
        assert elapsed >= SENTENCE_GUARD + SENTENCE_WINDOW - 0.05
        assert mouth.window_open is False  # 结束后关闭

    def test_window_is_open_during_listen_phase(self, mouth):
        """窗口期中途检查：window_open 应为 True（耳据此拾音）。"""
        import threading

        seen = []
        threading.Timer(SENTENCE_GUARD + 0.1, lambda: seen.append(mouth.window_open)).start()
        mouth._sentence_gap()
        assert seen == [True]

    def test_wires_sentence_done_callback(self):
        """后端有 sentence_done_callback 属性时，Mouth 注入句间钩子。"""

        class BackendWithHook:
            def __init__(self):
                self.sentence_done_callback = None

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

        b = BackendWithHook()
        m = Mouth(tts=b)
        assert b.sentence_done_callback == m._sentence_gap  # 绑定方法用 ==（每次访问是新对象）
