from PIL import Image

from backends import ToolCall
from backends.astrbot import phash, phash_score
from brain import pack_msg, parse_tool_args


def test_pack_msg_text():
    assert pack_msg("user", "text", "你好") == {"role": "user", "content": "你好"}


def test_pack_msg_image():
    msg = pack_msg("user", "image_url", "data:image/png;base64,xxx")
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][0]["image_url"]["url"] == "data:image/png;base64,xxx"


def test_pack_msg_tool():
    tool_call = ToolCall(id="call_1", name="look_at_screen", arguments="{}")
    msg = pack_msg("tool", "tool", "已完成", tool_call)
    assert msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "look_at_screen",
        "content": "已完成",
    }


def test_parse_tool_args_normal():
    assert parse_tool_args('{"a": 1}') == {"a": 1}


def test_parse_tool_args_empty():
    assert parse_tool_args("") == {}
    assert parse_tool_args(None) == {}


def test_parse_tool_args_null_and_non_dict():
    assert parse_tool_args("null") == {}
    assert parse_tool_args("[1, 2]") == {}


def test_parse_tool_args_garbage():
    assert parse_tool_args("这不是json") == {}


class TestPhash:
    """感知哈希：用于屏幕变化检测。注意纯色图是退化情形（哈希恒为全 1），用例用渐变图。"""

    @staticmethod
    def _gradient(shift=0):
        """L 模式渐变图：像素值 (x + y + shift) % 256，不同 shift 对应不同画面。"""
        img = Image.new("L", (320, 200))
        img.putdata([(x + y + shift) % 256 for y in range(200) for x in range(320)])
        return img

    def test_same_image_score_zero(self):
        img = self._gradient(shift=10)
        assert phash_score(phash(img), phash(img)) == 0.0

    def test_different_images_score_positive(self):
        score = phash_score(phash(self._gradient(0)), phash(self._gradient(128)))
        assert 0.0 < score <= 1.0

    def test_slight_change_low_score(self):
        """局部小改（一块区域亮度 +5）的变化幅度应低于整体换色。"""
        base = self._gradient(0)
        slight = base.copy()
        for y in range(100, 140):
            for x in range(100, 140):
                slight.putpixel((x, y), (x + y + 5) % 256)
        big = self._gradient(96)  # 整屏色相偏移
        assert phash_score(phash(base), phash(slight)) < phash_score(phash(base), phash(big))
