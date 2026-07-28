import asyncio
from types import SimpleNamespace

import pytest

from brain import MAX_FACTS, Brain, parse_fact_ops
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


class _StubClient:
    """假 LLM:固定回复 self.reply（测试中可随时改），无工具调用。"""

    def __init__(self, reply="好的"):
        self.reply = reply
        self.chat = SimpleNamespace(completions=self)

    async def create(self, model, messages, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=self.reply, tool_calls=None))]
        )


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


class TestFacts:
    def test_facts_crud(self, store):
        store.add_fact("用户在 UESTC 读书")
        store.add_fact("用户讨厌香菜")
        facts = store.get_facts()
        assert [c for _, c in facts] == ["用户在 UESTC 读书", "用户讨厌香菜"]
        store.update_fact(facts[1][0], "用户不吃香菜")
        store.delete_fact(facts[0][0])
        assert store.get_facts() == [(facts[1][0], "用户不吃香菜")]

    def test_meta_roundtrip(self, store):
        assert store.get_meta("k") is None
        store.set_meta("k", "1")
        store.set_meta("k", "2")
        assert store.get_meta("k") == "2"

    def test_get_turns_since_limit(self, store):
        for turn in range(1, 6):
            store.add_message(turn, _user_msg(f"第{turn}轮"))
        turn_ids, msgs = store.get_turns_since(2, limit=2)
        assert turn_ids == [4, 5]
        assert [m["content"] for m in msgs] == ["第4轮", "第5轮"]

    def test_parse_fact_ops(self):
        assert parse_fact_ops('{"operations": [{"op": "add", "content": "a"}]}') == [{"op": "add", "content": "a"}]
        assert parse_fact_ops('```json\n{"operations": []}\n```') == []
        assert parse_fact_ops("这不是json") == []
        assert parse_fact_ops('{"operations": "not a list"}') == []
        assert parse_fact_ops('{"operations": [{"op": "add"}, "junk", null]}') == [{"op": "add"}]

    def test_extract_adds_fact_and_advances_cursor(self, brain):
        brain.client.reply = '{"operations": [{"op": "add", "content": "用户在 UESTC 读书"}]}'

        async def run():
            await brain.get_llm_response("我在 UESTC 读书")
            await brain.maybe_extract_facts()
            # 没有新轮次时不重复抽取、不重复落库
            await brain.maybe_extract_facts()

        asyncio.run(run())
        assert [c for _, c in brain.memory.get_facts()] == ["用户在 UESTC 读书"]
        assert brain.memory.get_meta("last_extracted_turn") == str(brain.turn_id)

    def test_extract_bad_output_discarded(self, brain):
        brain.client.reply = "胡说八道"

        async def run():
            await brain.get_llm_response("你好")
            await brain.maybe_extract_facts()

        asyncio.run(run())
        assert brain.memory.get_facts() == []
        # 游标照常推进，不会反复重试同一批
        assert brain.memory.get_meta("last_extracted_turn") == str(brain.turn_id)

    def test_extract_covers_memorized_turns(self, brain):
        brain.memorize("我下周三要交实验报告")
        brain.client.reply = '{"operations": [{"op": "add", "content": "用户下周三要交实验报告"}]}'

        async def run():
            await brain.get_llm_response("糯糯你在吗")
            await brain.maybe_extract_facts()

        asyncio.run(run())
        assert [c for _, c in brain.memory.get_facts()] == ["用户下周三要交实验报告"]

    def test_apply_fact_ops_validation(self, brain):
        brain.memory.add_fact("旧事实")
        fid = brain.memory.get_facts()[0][0]
        brain._apply_fact_ops(
            [
                {"op": "update", "id": fid, "content": "新事实"},
                {"op": "update", "id": 999, "content": "不存在的 id"},
                {"op": "add", "content": "   "},
                {"op": "delete", "id": fid},
                {"op": "unknown"},
            ]
        )
        assert brain.memory.get_facts() == []

    def test_facts_injected_into_messages(self, brain):
        brain.memory.add_fact("用户叫小明")
        messages = brain._build_messages()
        assert any("用户叫小明" in m["content"] for m in messages[:2])

    def test_merge_facts_over_limit(self, brain):
        for i in range(MAX_FACTS + 1):
            brain.memory.add_fact(f"事实{i}")
        brain.client.reply = '["合并事实1", "合并事实2"]'
        asyncio.run(brain._maybe_merge_facts())
        assert [c for _, c in brain.memory.get_facts()] == ["合并事实1", "合并事实2"]

    def test_merge_bad_output_keeps_original(self, brain):
        for i in range(MAX_FACTS + 1):
            brain.memory.add_fact(f"事实{i}")
        brain.client.reply = "不是json"
        asyncio.run(brain._maybe_merge_facts())
        assert len(brain.memory.get_facts()) == MAX_FACTS + 1
