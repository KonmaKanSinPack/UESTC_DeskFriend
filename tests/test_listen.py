"""Listen 回声门控测试（纯逻辑，不实例化 Listen 以免开麦克风流/加载模型）。

门控规则（句间监听窗口方案）：VAD 触发时，嘴发声且监听窗口未开（句子播放中/
余响静默期）→ 视为回声丢弃；窗口期触发 → 用户说话（打断）。
纯函数提取照 astrbot.py 的 should_observe 模式（便于单测），录音主循环只在关键点调用。
"""

import numpy as np

from listen import (
    PreRollBuffer,
    boost_if_quiet,
    segment_contaminated,
    should_drop_echo,
    trim_trailing_silence,
    window_interrupt_confirmed,
)


class TestWindowInterruptConfirmed:
    """窗口期打断确认：最近 10 块活跃 ≥8 才算用户说话（余响是单发尖峰）。"""

    def test_sustained_speech_confirmed(self):
        assert window_interrupt_confirmed([True] * 8) is True
        assert window_interrupt_confirmed([True] * 10) is True

    def test_ring_burst_not_confirmed(self):
        """余响尖峰（实测 ~5 块）够不着阈值。"""
        assert window_interrupt_confirmed([True] * 5 + [False] * 5) is False

    def test_mixed_activity_below_threshold(self):
        assert window_interrupt_confirmed([True, False] * 5) is False

    def test_empty_not_confirmed(self):
        assert window_interrupt_confirmed([]) is False


class TestShouldDropEcho:
    def test_not_speaking_listens(self):
        """嘴未发声：正常监听（无论窗口状态）。"""
        assert should_drop_echo(False, False) is False
        assert should_drop_echo(False, True) is False

    def test_speaking_window_closed_drops(self):
        """句子播放中 / 余响静默期（窗口未开）→ 拾到的必是扬声器回声，丢弃。"""
        assert should_drop_echo(True, False) is True

    def test_speaking_window_open_listens(self):
        """句间监听窗口 → 拾音（触发即用户说话）。"""
        assert should_drop_echo(True, True) is False


class TestSegmentContaminated:
    def test_no_overlap_is_clean(self):
        assert segment_contaminated([False, False, False]) is False

    def test_any_overlap_contaminated(self):
        assert segment_contaminated([False, True, False]) is True
        assert segment_contaminated([True]) is True

    def test_empty_segment_clean(self):
        assert segment_contaminated([]) is False


class TestSpeakingSessionResetsLatches:
    """新一段朗读（speaking False→True）→ 清打断闩/冷却，防上一段冷却抑制本段首窗。"""

    def test_new_speaking_session_resets_interrupt_latches(self):
        import types
        from collections import deque

        from listen import WINDOW_ACTIVITY_HISTORY, Listen

        lis = Listen.__new__(Listen)  # 绕过 __init__（不开麦克风 / 不加载模型）；QObject 需用类自身 __new__
        lis.mouth = types.SimpleNamespace(speaking=True, window_open=False)
        lis._prev_speaking = False
        lis._prev_window_open = False
        lis._interrupt_sent = True  # 上一段遗留
        lis._interrupt_cooldown_until = 9e9  # 上一段遗留冷却（远未到期）
        lis._window_activity = deque([True] * 5, maxlen=WINDOW_ACTIVITY_HISTORY)
        lis._window_tick(0.9)
        assert lis._interrupt_sent is False
        assert lis._interrupt_cooldown_until == 0.0
        assert len(lis._window_activity) == 0


# ---------- 音频三小修（2026-09-03） ----------


class TestPreRollBuffer:
    def test_drain_returns_chunks_and_flags_in_order(self):
        buf = PreRollBuffer()
        buf.append("a", False)
        buf.append("b", True)
        chunks, playing = buf.drain()
        assert chunks == ["a", "b"]
        assert playing == [False, True]

    def test_drain_clears(self):
        buf = PreRollBuffer()
        buf.append("a", False)
        buf.drain()
        assert buf.drain() == ([], [])

    def test_capacity_keeps_latest(self):
        buf = PreRollBuffer(max_chunks=2)
        for i in range(4):
            buf.append(f"c{i}", False)
        chunks, _ = buf.drain()
        assert chunks == ["c2", "c3"]  # 只留最近 2 块（环形语义）


class TestTrimTrailingSilence:
    def test_strips_tail_keeps_pause(self):
        chunks = ["s1", "s2", "q1", "q2", "q3", "q4"]
        scores = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
        # 尾部 4 块静音裁掉，保留 3 块停顿 → 到 s2 后 3 块（q1..q3）
        assert trim_trailing_silence(chunks, scores) == ["s1", "s2", "q1", "q2", "q3"]

    def test_all_silent_keeps_min(self):
        chunks = ["a", "b"]
        scores = [0.1, 0.1]
        assert trim_trailing_silence(chunks, scores, keep=1) == ["a"]

    def test_loud_tail_untouched(self):
        chunks = ["a", "b"]
        scores = [0.1, 0.9]
        assert trim_trailing_silence(chunks, scores) == ["a", "b"]


class TestBoostIfQuiet:
    def test_quiet_audio_scaled_up(self):
        audio = np.ones(100, dtype=np.float32) * 0.1  # 峰值 0.1 < 0.25
        boosted = boost_if_quiet(audio)
        assert abs(float(np.max(boosted)) - 0.5) < 1e-6
        assert abs(boosted[0] - 0.5) < 1e-6  # 等比放大，波形不变

    def test_normal_audio_untouched(self):
        audio = np.ones(100, dtype=np.float32) * 0.7
        assert boost_if_quiet(audio) is audio  # 正常音量原样返回（不复制不放大）

    def test_silence_untouched(self):
        audio = np.zeros(100, dtype=np.float32)
        assert boost_if_quiet(audio) is audio  # 全静音不放大（防抬底噪）
