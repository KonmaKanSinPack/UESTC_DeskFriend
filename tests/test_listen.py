"""Listen 回声门控测试（纯逻辑，不实例化 Listen 以免开麦克风流/加载模型）。

门控规则：TTS 朗读期间麦克风拾到的语音视为扬声器回声——若当真会触发
on_heard_text → interrupt 打断自己（自反馈回环）。纯函数提取照 astrbot.py
的 should_observe 模式（便于单测），录音主循环只在关键点调用。
"""

from listen import is_echo_trigger, segment_contaminated


class TestEchoTrigger:
    def test_vad_trigger_during_speaking_is_echo(self):
        assert is_echo_trigger(0.9, speaking=True) is True

    def test_vad_trigger_when_silent_is_user_speech(self):
        assert is_echo_trigger(0.9, speaking=False) is False

    def test_low_score_never_echo(self):
        assert is_echo_trigger(0.3, speaking=True) is False

    def test_score_boundary_inclusive(self):
        # VAD 判定阈值为 >= 0.5，回声判定保持一致（边界含等号）
        assert is_echo_trigger(0.5, speaking=True) is True


class TestSegmentContaminated:
    def test_no_overlap_is_clean(self):
        assert segment_contaminated([False, False, False]) is False

    def test_any_overlap_contaminated(self):
        assert segment_contaminated([False, True, False]) is True
        assert segment_contaminated([True]) is True

    def test_empty_segment_clean(self):
        assert segment_contaminated([]) is False
