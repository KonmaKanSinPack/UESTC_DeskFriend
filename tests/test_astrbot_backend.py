"""AstrBotBackend 行为测试：注入假 bridge，验证消息转换、屏幕关键词附图、
[look_at_screen] 协议、兜底文案、唤醒规则、主动观察冒泡与门控逻辑。
"""

import asyncio

import pytest

from backends.astrbot import (
    PROACTIVE_OBSERVE_TEXT,
    AstrBotBackend,
    _is_no_reply,
    clean_markdown,
    should_observe,
)
from brain import pack_msg

FAKE_IMG = {"type": "image", "data": {"file": "base64://AAAA", "subType": 0}}


class FakeBridge:
    """假 OneBot 通道：记录发出的 segments，按序吐出预设回复。"""

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.sent = []

    def start(self):
        pass

    async def ensure_connected(self):
        pass

    @property
    def busy(self):
        return False

    async def send_message(self, segments):
        self.sent.append(segments)
        return self.replies.pop(0) if self.replies else ""


class FakeJudge:
    """假决策器：按预设列表依次返回判定结果。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [])
        self.called_with = []

    async def should_reply(self, user_text):
        self.called_with.append(user_text)
        return self.replies.pop(0) if self.replies else False


@pytest.fixture()
def backend():
    return AstrBotBackend(url="ws://fake", bridge=FakeBridge(), enable_observer=False)


@pytest.fixture()
def backend_with_judge():
    return AstrBotBackend(url="ws://fake", bridge=FakeBridge(), judge=FakeJudge(), enable_observer=False)


class TestConversation:
    def test_text_message_sent_and_reply(self, backend):
        backend.bridge.replies = ["你好呀"]
        resp = asyncio.run(backend.get_llm_response("你好"))
        assert resp.content == "你好呀"
        assert resp.tool_calls == []
        assert backend.bridge.sent == [[{"type": "text", "data": {"text": "你好"}}]]

    def test_interruption_marker_injected_and_cleared(self, backend):
        """TTS 朗读被打断 → 打断位置注入对话消息（让桃桃知道说到哪），用后清除。"""
        backend.interruption = "你刚才说到第一句"
        backend.bridge.replies = ["好，你说"]
        resp = asyncio.run(backend.get_llm_response("我继续说"))
        assert "对话被打断" in backend.bridge.sent[0][0]["data"]["text"]
        assert "你刚才说到第一句" in backend.bridge.sent[0][0]["data"]["text"]
        assert "我继续说" in backend.bridge.sent[0][0]["data"]["text"]
        assert backend.interruption is None  # 用后清除
        assert resp.content == "好，你说"

    def test_no_interruption_marker_when_clean(self, backend):
        backend.bridge.replies = ["好"]
        asyncio.run(backend.get_llm_response("正常消息"))
        assert "对话被打断" not in backend.bridge.sent[0][0]["data"]["text"]

    def test_image_message_converted(self, backend):
        backend.bridge.replies = ["看到了"]
        img_msg = pack_msg("user", "image_url", "data:image/png;base64,BBBB")
        asyncio.run(backend.get_llm_response(img_msg))
        assert backend.bridge.sent == [[{"type": "image", "data": {"file": "base64://BBBB", "subType": 0}}]]

    def test_screen_keyword_attaches_capture(self, backend, monkeypatch):
        """命中屏幕关键词 → 本地截屏，文字+图片一并发桃桃。"""
        monkeypatch.setattr(backend, "_capture_segment", lambda: FAKE_IMG)
        backend.bridge.replies = ["你在写代码呢"]
        asyncio.run(backend.get_llm_response("看看我的屏幕"))
        assert backend.bridge.sent == [[{"type": "text", "data": {"text": "看看我的屏幕"}}, FAKE_IMG]]

    def test_tool_message_skipped(self, backend):
        """tool 消息在 OneBot 通道无法表达 → 跳过，只剩兜底文本。"""
        backend.bridge.replies = [""]
        tool_msg = pack_msg("tool", "tool", "已完成", __import__("types").SimpleNamespace(id="c1", name="t"))
        resp = asyncio.run(backend.get_llm_response([tool_msg]))
        assert backend.bridge.sent == [[{"type": "text", "data": {"text": "…"}}]]
        assert resp.content == "（桃桃没有回应…）"

    def test_no_reply_fallback(self, backend):
        backend.bridge.replies = [""]
        resp = asyncio.run(backend.get_llm_response("在吗"))
        assert resp.content == "（桃桃没有回应…）"


class TestLookAtScreenProtocol:
    def test_marker_triggers_capture_followup(self, backend, monkeypatch):
        """桃桃回复含 [look_at_screen] → 截屏附图追问 → 返回最终回复（标记剥离）。"""
        monkeypatch.setattr(backend, "_capture_segment", lambda: FAKE_IMG)
        backend.bridge.replies = ["让我看看[look_at_screen]", "我看到你在写代码"]
        resp = asyncio.run(backend.get_llm_response("我现在在干嘛"))
        assert resp.content == "我看到你在写代码"
        assert len(backend.bridge.sent) == 2
        assert backend.bridge.sent[1] == [{"type": "text", "data": {"text": "（这是你刚才要看的屏幕）"}}, FAKE_IMG]

    def test_capture_failure_keeps_reply(self, backend, monkeypatch):
        """截图失败（后端未就绪）：保留带标记的回复原文，不再追问。"""
        monkeypatch.setattr(backend, "_capture_segment", lambda: None)
        backend.bridge.replies = ["让我看看[look_at_screen]"]
        resp = asyncio.run(backend.get_llm_response("看下屏幕"))
        assert resp.content == "让我看看"  # 标记剥离后原文展示
        assert len(backend.bridge.sent) == 1


class TestLifecycle:
    def test_constructs_before_loop_running(self):
        """回归：qasync 时序——loop 已 set 但未 run 时构造后端不崩溃。

        main.py 在 `loop.run_forever()` 之前就构造 DeskFriend（内部创建 Brain），
        后台任务必须挂在"已设置但未运行"的 loop 上；
        曾用 get_running_loop() → RuntimeError: no running event loop（真机启动崩溃）。
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            backend = AstrBotBackend(url="ws://127.0.0.1:1", enable_observer=True)
            # 构造不抛错即通过
        finally:
            loop.run_until_complete(backend.stop())
            loop.close()
            asyncio.set_event_loop(None)


class TestWakeRules:
    """should_reply 判定统一走 LLM 决策器（不用本地关键词）。"""

    def test_judge_true(self, backend_with_judge):
        backend_with_judge.judge.replies = [True]
        judge = [{"role": "user", "content": "用户的消息是：糯糯你在吗"}]
        resp = asyncio.run(backend_with_judge.get_response_with_context(judge))
        assert resp.content == "true"

    def test_judge_false(self, backend_with_judge):
        backend_with_judge.judge.replies = [False]
        judge = [{"role": "user", "content": "用户的消息是：嗯嗯"}]
        resp = asyncio.run(backend_with_judge.get_response_with_context(judge))
        assert resp.content == "false"

    def test_judge_receives_stripped_message(self, backend_with_judge):
        """ui 的"用户的消息是："前缀必须剥离，决策器只看到用户原文。"""
        backend_with_judge.judge.replies = [True]
        judge = [{"role": "user", "content": "用户的消息是：今天好累啊"}]
        asyncio.run(backend_with_judge.get_response_with_context(judge))
        assert backend_with_judge.judge.called_with == ["今天好累啊"]

    def test_empty_context_judged_as_empty_text(self, backend_with_judge):
        backend_with_judge.judge.replies = [False]
        resp = asyncio.run(backend_with_judge.get_response_with_context([]))
        assert resp.content == "false"
        assert backend_with_judge.judge.called_with == [""]

    def test_no_judge_defaults_to_reply(self, backend):
        """判定器未配置（缺 API_KEY）→ 默认回复，不抛错。"""
        judge = [{"role": "user", "content": "用户的消息是：在吗"}]
        resp = asyncio.run(backend.get_response_with_context(judge))
        assert resp.content == "true"


class TestMemoryNoop:
    def test_memory_ops_are_noops(self, backend):
        """记忆由 AstrBot 接管：三接口空操作，不抛错。"""
        backend.memorize("背景谈话")
        asyncio.run(backend.maybe_compress())
        asyncio.run(backend.maybe_extract_facts())


class TestProactiveLook:
    def test_sink_called_on_worthwhile_reply(self, backend, monkeypatch):
        monkeypatch.setattr(backend, "_capture_segment", lambda: FAKE_IMG)
        bubbles = []
        backend.reply_sink = bubbles.append
        backend.bridge.replies = ["主人，你屏幕上出现了新通知！"]
        asyncio.run(backend._proactive_look())
        assert bubbles == ["主人，你屏幕上出现了新通知！"]
        assert backend.bridge.sent[0][0]["data"]["text"] == PROACTIVE_OBSERVE_TEXT

    def test_sink_silent_on_no_reply(self, backend, monkeypatch):
        monkeypatch.setattr(backend, "_capture_segment", lambda: FAKE_IMG)
        bubbles = []
        backend.reply_sink = bubbles.append
        backend.bridge.replies = ["无"]
        asyncio.run(backend._proactive_look())
        assert bubbles == []

    def test_sink_silent_without_sink(self, backend, monkeypatch):
        monkeypatch.setattr(backend, "_capture_segment", lambda: FAKE_IMG)
        backend.bridge.replies = ["有话说"]
        asyncio.run(backend._proactive_look())  # 无 sink 不抛错


class TestGate:
    def test_gate_allows_when_everything_passes(self):
        assert should_observe(
            change_score=0.3,
            now=1000,
            last_proactive_at=0,
            cooldown=180,
            busy=False,
            last_user_at=0,
            user_silence=120,
            threshold=0.05,
        )

    def test_gate_blocks_each_condition(self):
        base = dict(
            change_score=0.3,
            now=1000,
            last_proactive_at=0,
            cooldown=180,
            busy=False,
            last_user_at=0,
            user_silence=120,
            threshold=0.05,
        )
        assert not should_observe(**{**base, "busy": True})  # 对话中
        assert not should_observe(**{**base, "change_score": 0.01})  # 画面没变
        assert not should_observe(**{**base, "last_proactive_at": 900})  # 冷却中
        assert not should_observe(**{**base, "last_user_at": 950})  # 用户操作中


class TestCleanMarkdown:
    def test_code_fences(self):
        assert clean_markdown("```x\n内容\n```") == "内容"
        assert clean_markdown("```(๑•̀ㅁ•́)و✧`\n正文\n```(・_・;`") == "正文"

    def test_emphasis(self):
        assert clean_markdown("**加粗** 和 *斜体*") == "加粗 和 斜体"

    def test_heading_and_inline_code(self):
        assert clean_markdown("# 标题\n`code`") == "标题\ncode"

    def test_plain_text_untouched(self):
        assert clean_markdown("普通文本，没有符号") == "普通文本，没有符号"
        assert clean_markdown("") == ""


class TestIsNoReply:
    def test_no_reply_variants(self):
        assert _is_no_reply("无")
        assert _is_no_reply("没有")
        assert _is_no_reply("没什么")
        assert _is_no_reply("")
        assert _is_no_reply("。")  # 标点算空

    def test_real_reply_not_discarded(self):
        assert not _is_no_reply("屏幕上有新消息提醒")
        assert not _is_no_reply("无伤大雅")  # 含"无"但不是表示无话可说
