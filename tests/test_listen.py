"""Listen 回声门控测试（纯逻辑，不实例化 Listen 以免开麦克风流/加载模型）。

门控规则（AEC 轮）：VAD 触发时按「嘴是否发声 + AEC 是否就绪且收敛」三分支决策——
收敛期（或无 AEC）拾到的语音视为回声丢弃；AEC 收敛后残差触发 = 真人插嘴。
纯函数提取照 astrbot.py 的 should_observe 模式（便于单测），录音主循环只在关键点调用。
"""

from listen import echo_gate_action, segment_contaminated


class TestEchoGateAction:
    def test_not_speaking_normal(self):
        """嘴未发声：正常录音（无论 AEC 状态）。"""
        assert echo_gate_action(False, True, True) is None
        assert echo_gate_action(False, False, False) is None

    def test_speaking_without_aec_drops(self):
        """AEC 不可用 → 回退纯门控：发声期间一律丢弃。"""
        assert echo_gate_action(True, False, False) == "drop"

    def test_speaking_during_convergence_drops(self):
        """AEC 收敛期内 → 仍丢弃（回声可能漏过，旧门控兜底）。"""
        assert echo_gate_action(True, True, False) == "drop"

    def test_speaking_after_convergence_barges_in(self):
        """AEC 已收敛 → 残差触发 = 真人插嘴。"""
        assert echo_gate_action(True, True, True) == "barge_in"


class TestSegmentContaminated:
    def test_no_overlap_is_clean(self):
        assert segment_contaminated([False, False, False]) is False

    def test_any_overlap_contaminated(self):
        assert segment_contaminated([False, True, False]) is True
        assert segment_contaminated([True]) is True

    def test_empty_segment_clean(self):
        assert segment_contaminated([]) is False
