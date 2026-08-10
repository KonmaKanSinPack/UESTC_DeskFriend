from PIL import Image

from backends import ToolCall
from backends.astrbot import phash, phash_score, should_reply_local
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


class TestShouldReplyLocal:
    """本地唤醒规则（反向判定）：名字/指令/问候/疑问/触碰彩蛋/主动搭话 → True；纯语气词碎片 → False。"""

    def test_called_by_name(self):
        assert should_reply_local("糯糯你在吗")
        assert should_reply_local("桃桃，过来")

    def test_direct_commands(self):
        for text in ("看看我的屏幕", "帮我截个图", "打开桌面", "理我一下"):
            assert should_reply_local(text), text

    def test_greetings(self):
        assert should_reply_local("你好呀")
        assert should_reply_local("早安")

    def test_questions(self):
        assert should_reply_local("今天天气怎么样？")
        assert should_reply_local("现在几点了")
        assert should_reply_local("为什么月亮是圆的")
        assert should_reply_local("帮我看下这个")

    def test_poke_interaction(self):
        assert should_reply_local("用户用鼠标触碰了你")

    def test_background_noise(self):
        # 纯语气词/填充词 → 不打扰
        for t in ("嗯嗯", "好", "好的", "哈哈", "知道了", "哦哦", "ok", "嗯嗯嗯"):
            assert not should_reply_local(t), t
        assert not should_reply_local("")

    def test_active_talk_default_reply(self):
        """主动搭话没有名单关键词也回（宁可多回不可漏听，治漏判）。"""
        for t in ("陪我玩", "讲个笑话", "好无聊啊", "给我唱首歌", "说点有意思的", "今天有什么新鲜事"):
            assert should_reply_local(t), t


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
