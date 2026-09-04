"""PsdRenderer 纯数学单测：眨眼曲线/视线钳制/口型振荡（不实例化 QWidget）。"""

from psd_renderer import blink_scale, clamp, eye_track_offset, mouth_scale


class TestBlinkScale:
    def test_rest_states_are_open(self):
        assert blink_scale(-0.1) == 1.0  # 未开始
        assert blink_scale(0.5) == 1.0  # 已结束

    def test_midpoint_is_near_closed(self):
        mid = blink_scale(0.13 / 2)  # 半程
        assert mid < 0.15  # 谷底 ~0.05（留一条缝更像眯眼）

    def test_curve_continuous(self):
        """起止两侧与曲线衔接（无跳变）：接近边界的值接近 1。"""
        assert blink_scale(0.01) > 0.7
        assert blink_scale(0.12) > 0.7


class TestEyeTrackOffset:
    def test_far_target_full_range(self):
        dx, dy = eye_track_offset(1000.0, 0.0, 1000.0)
        assert abs(dx - 14.0) < 0.01 and abs(dy) < 0.01  # 满幅向右

    def test_near_target_scaled_down(self):
        dx, dy = eye_track_offset(40.0, 0.0, 40.0)
        assert 0 < dx < 14.0  # 贴近时幅度收敛（不至于斗鸡眼）

    def test_zero_vector_safe(self):
        assert eye_track_offset(0.0, 0.0, 0.0) == (0.0, 0.0)


class TestMouthScale:
    def test_not_talking_closed(self):
        assert mouth_scale(123.0, talking=False) == 1.0

    def test_talking_oscillates(self):
        period = 1.0 / 6.0
        vals = [mouth_scale(t, talking=True) for t in (0.0, period / 8, period / 4, period / 2)]
        assert vals[0] == 1.0  # 起点闭合
        assert vals[1] > 1.3  # 八分之一周期已张开
        assert vals[2] > 1.4  # 四分之一周期峰值（全开）
        assert abs(vals[3] - 1.0) < 0.05  # 半周期回闭合


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10
