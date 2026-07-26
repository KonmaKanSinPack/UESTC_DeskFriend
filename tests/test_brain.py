from types import SimpleNamespace

from brain import pack_msg, parse_tool_args


def test_pack_msg_text():
    assert pack_msg("user", "text", "你好") == {"role": "user", "content": "你好"}


def test_pack_msg_image():
    msg = pack_msg("user", "image_url", "data:image/png;base64,xxx")
    assert msg["role"] == "user"
    assert msg["content"][0]["type"] == "image_url"
    assert msg["content"][0]["image_url"]["url"] == "data:image/png;base64,xxx"


def test_pack_msg_tool():
    tool_call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="look_at_screen"))
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
