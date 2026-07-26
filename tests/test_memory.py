import asyncio
from types import SimpleNamespace

import pytest

from brain import Brain
from memory import KEEP_RECENT_TURNS, MAX_CONTEXT_TURNS, MemoryStore, render_messages, sanitize_message


@pytest.fixture()
def store(tmp_path):
    return MemoryStore(tmp_path / "pet.db")


def _user_msg(text):
    return {"role": "user", "content": text}


class TestMemoryStore:
    def test_add_and_load_roundtrip(self, store):
        store.add_message(1, _user_msg("你好"))
        store.add_message(1, {"role": "assistant", "content": "你好呀"})
        store.add_message(2, _user_msg("我叫小明"))
        msgs = store.load_unsummarized()
        assert [m["content"] for m in msgs] == ["你好", "你好呀", "我叫小明"]

    def test_turn_id_monotonic(self, store):
        assert store.next_turn_id() == 1
        store.add_message(1, _user_msg("a"))
        store.add_message(2, _user_msg("b"))
        assert store.max_turn_id() == 2
        assert store.next_turn_id() == 3

    def test_unsummarized_turn_count(self, store):
        for turn in range(1, 4):
            store.add_message(turn, _user_msg(f"第{turn}轮"))
            store.add_message(turn, {"role": "assistant", "content": "好"})
        assert store.unsummarized_turn_count() == 3

    def test_turns_to_summarize_keeps_recent(self, store):
        for turn in range(1, 6):
            store.add_message(turn, _user_msg(f"第{turn}轮"))
        turn_ids, msgs = store.turns_to_summarize(keep_recent=2)
        assert turn_ids == [1, 2, 3]
        assert [m["content"] for m in msgs] == ["第1轮", "第2轮", "第3轮"]

    def test_mark_summarized(self, store):
        for turn in range(1, 4):
            store.add_message(turn, _user_msg(f"第{turn}轮"))
        store.mark_summarized([1, 2])
        assert store.unsummarized_turn_count() == 1
        assert [m["content"] for m in store.load_unsummarized()] == ["第3轮"]

    def test_summary_upsert(self, store):
        assert store.get_summary() is None
        store.set_summary("第一版摘要")
        store.set_summary("更新后的摘要")
        assert store.get_summary() == "更新后的摘要"


class TestSanitize:
    def test_image_replaced(self):
        msg = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }
        clean = sanitize_message(msg)
        assert clean["content"][0]["image_url"]["url"] == "[image omitted]"
        # 原消息不被修改
        assert msg["content"][0]["image_url"]["url"] == "data:image/png;base64,AAAA"

    def test_plain_text_untouched(self):
        assert sanitize_message(_user_msg("你好")) == _user_msg("你好")

    def test_render_messages(self):
        msgs = [
            _user_msg("看看我的屏幕"),
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "name": "look_at_screen", "content": "已查看"},
            {"role": "assistant", "content": "你在写代码"},
        ]
        text = render_messages(msgs)
        assert "用户: 看看我的屏幕" in text
        assert "[工具 look_at_screen] 已查看" in text
        assert "糯糯: 你在写代码" in text


class _StubCompletions:
    """假 LLM:固定回复一句话，无工具调用。"""

    async def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="好的", tool_calls=None))]
        )


class _StubClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_StubCompletions())


@pytest.fixture()
def brain(tmp_path):
    return Brain(client=_StubClient(), db_path=tmp_path / "pet.db")


class TestBrainMemory:
    def test_messages_persisted_with_turns(self, brain):
        async def run():
            await brain.get_llm_response("你好")
            await brain.get_llm_response("我叫小明")

        asyncio.run(run())
        msgs = brain.memory.load_unsummarized()
        assert [m["content"] for m in msgs] == ["你好", "好的", "我叫小明", "好的"]
        assert brain.memory.unsummarized_turn_count() == 2

    def test_restart_restores_context(self, brain, tmp_path):
        asyncio.run(brain.get_llm_response("记住我叫小明"))
        # 模拟重启：同一个 db 新建 Brain
        brain2 = Brain(client=_StubClient(), db_path=tmp_path / "pet.db")
        assert [m["content"] for m in brain2.context] == ["记住我叫小明", "好的"]
        assert brain2.turn_id == brain.turn_id

    def test_compress_not_triggered_below_limit(self, brain):
        async def run():
            await brain.get_llm_response("你好")
            await brain.maybe_compress()

        asyncio.run(run())
        assert brain.summary is None
        assert len(brain.context) == 2

    def test_compress_merges_old_turns(self, brain):
        async def run():
            for i in range(MAX_CONTEXT_TURNS + 1):
                await brain.get_llm_response(f"第{i + 1}轮")
            await brain.maybe_compress()

        asyncio.run(run())
        # 摘要已更新，旧轮次被标记压缩，内存只留最近 KEEP_RECENT_TURNS 轮（每轮 user+assistant 两条）
        assert brain.summary == "好的"
        assert brain.memory.unsummarized_turn_count() == KEEP_RECENT_TURNS
        assert len(brain.context) == KEEP_RECENT_TURNS * 2
        assert brain.context[0]["content"] == f"第{MAX_CONTEXT_TURNS + 2 - KEEP_RECENT_TURNS}轮"

    def test_summary_injected_into_messages(self, brain):
        brain.summary = "用户叫小明"
        messages = brain._build_messages()
        assert messages[0]["role"] == "system"
        assert "用户叫小明" in messages[1]["content"]

    def test_memorize_stores_without_reply(self, brain):
        brain.memorize("我下周三要交实验报告")
        msgs = brain.memory.load_unsummarized()
        assert len(msgs) == 1
        assert "我下周三要交实验报告" in msgs[0]["content"]
        assert "[背景谈话" in msgs[0]["content"]
        assert brain.memory.unsummarized_turn_count() == 1

    def test_memorize_then_reply_keeps_order_and_turns(self, brain):
        brain.memorize("背景谈话一句")
        asyncio.run(brain.get_llm_response("糯糯你在吗"))
        contents = [m["content"] for m in brain.context]
        assert contents == ["[背景谈话，无需回应] 背景谈话一句", "糯糯你在吗", "好的"]
        # 背景记忆占独立轮次，后续回复开启新一轮
        assert brain.memory.unsummarized_turn_count() == 2
