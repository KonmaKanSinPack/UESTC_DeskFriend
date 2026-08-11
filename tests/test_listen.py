"""Listen 回声门控测试（纯逻辑，不实例化 Listen 以免开麦克风流/加载模型）。

门控规则（句间监听窗口方案）：VAD 触发时，嘴发声且监听窗口未开（句子播放中/
余响静默期）→ 视为回声丢弃；窗口期触发 → 用户说话（打断）。
纯函数提取照 astrbot.py 的 should_observe 模式（便于单测），录音主循环只在关键点调用。
"""

from listen import segment_contaminated, should_drop_echo


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
