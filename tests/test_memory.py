import asyncio
from types import SimpleNamespace

import pytest

from backends.judger import JUDGE_SYSTEM_PROMPT
from backends.openai import (
    DEFAULT_KEEP_RECENT_TURNS as KEEP_RECENT_TURNS,
)
from backends.openai import (
    DEFAULT_MAX_CONTEXT_TURNS as MAX_CONTEXT_TURNS,
)
from backends.openai import (
    DEFAULT_MAX_FACTS as MAX_FACTS,
)
from backends.openai import (
    SUMMARY_SYSTEM_PROMPT,
    OpenAIBackend,
    parse_fact_ops,
)
from memory import MemoryStore, render_messages, revive_message, sanitize_message


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


class TestRevive:
    """库 → 运行时还原：图片占位符必须变成 API 安全的文本段。

    回归背景（2026-09-04）：占位符留在 image_url.url 里，重启恢复的上下文发给
    API 被 base64 解码报 500 convert_request_failed，之后每轮对话全挂。
    """

    def test_placeholder_image_becomes_text_part(self):
        msg = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "[image omitted]"}}],
        }
        revived = revive_message(msg)
        assert revived["content"] == [{"type": "text", "text": "[历史截图，内容已省略]"}]
        # 原消息不被修改（DB 里的占位符原样保留）
        assert msg["content"][0]["image_url"]["url"] == "[image omitted]"

    def test_real_data_url_preserved(self):
        msg = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }
        assert revive_message(msg) == msg  # 真实内存图片（data: 开头）原样通过

    def test_plain_text_untouched(self):
        assert revive_message(_user_msg("你好")) == _user_msg("你好")

    def test_store_roundtrip_is_api_safe(self, store):
        """全链路回归：带图消息落库（脱敏）→ 重启恢复 → 上下文里无非法 image_url。"""
        img_msg = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }
        store.add_message(1, img_msg)
        msgs = store.load_unsummarized()
        assert msgs[0]["content"] == [{"type": "text", "text": "[历史截图，内容已省略]"}]


class TestRenderMessages:
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
        assert "桃桃: 你在写代码" in text  # 助手显示名（用户 2026-09-04 由糯糯改为桃桃）


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
def backend(tmp_path):
    """openai 回复后端（原 Brain 直连模式的载体），注入假 LLM 客户端。"""
    return OpenAIBackend(client=_StubClient(), db_path=tmp_path / "pet.db")


class TestOpenAIBackend:
    """openai 后端的记忆管道（原 TestBrainMemory 迁移）：对话落库、重启恢复、摘要压缩、背景谈话。"""

    def test_messages_persisted_with_turns(self, backend):
        async def run():
            await backend.get_llm_response("你好")
            await backend.get_llm_response("我叫小明")

        asyncio.run(run())
        msgs = backend.memory.load_unsummarized()
        assert [m["content"] for m in msgs] == ["你好", "好的", "我叫小明", "好的"]
        assert backend.memory.unsummarized_turn_count() == 2

    def test_restart_restores_context(self, backend, tmp_path):
        asyncio.run(backend.get_llm_response("记住我叫小明"))
        # 模拟重启：同一个 db 新建后端
        backend2 = OpenAIBackend(client=_StubClient(), db_path=tmp_path / "pet.db")
        assert [m["content"] for m in backend2.context] == ["记住我叫小明", "好的"]
        assert backend2.turn_id == backend.turn_id

    def test_compress_not_triggered_below_limit(self, backend):
        async def run():
            await backend.get_llm_response("你好")
            await backend.maybe_compress()

        asyncio.run(run())
        assert backend.summary is None
        assert len(backend.context) == 2

    def test_compress_merges_old_turns(self, backend):
        async def run():
            for i in range(MAX_CONTEXT_TURNS + 1):
                await backend.get_llm_response(f"第{i + 1}轮")
            await backend.maybe_compress()

        asyncio.run(run())
        # 摘要已更新，旧轮次被标记压缩，内存只留最近 KEEP_RECENT_TURNS 轮（每轮 user+assistant 两条）
        assert backend.summary == "好的"
        assert backend.memory.unsummarized_turn_count() == KEEP_RECENT_TURNS
        assert len(backend.context) == KEEP_RECENT_TURNS * 2
        assert backend.context[0]["content"] == f"第{MAX_CONTEXT_TURNS + 2 - KEEP_RECENT_TURNS}轮"

    def test_summary_injected_into_messages(self, backend):
        backend.summary = "用户叫小明"
        messages = backend._build_messages()
        assert messages[0]["role"] == "system"
        assert "用户叫小明" in messages[1]["content"]

    def test_memorize_stores_without_reply(self, backend):
        backend.memorize("我下周三要交实验报告")
        msgs = backend.memory.load_unsummarized()
        assert len(msgs) == 1
        assert "我下周三要交实验报告" in msgs[0]["content"]
        assert "[背景谈话" in msgs[0]["content"]
        assert backend.memory.unsummarized_turn_count() == 1

    def test_memorize_then_reply_keeps_order_and_turns(self, backend):
        backend.memorize("背景谈话一句")
        asyncio.run(backend.get_llm_response("糯糯你在吗"))
        contents = [m["content"] for m in backend.context]
        assert contents == ["[背景谈话，无需回应] 背景谈话一句", "糯糯你在吗", "好的"]
        # 背景记忆占独立轮次，后续回复开启新一轮
        assert backend.memory.unsummarized_turn_count() == 2

    def test_response_normalized(self, backend):
        """返回值归一为 BackendResponse（无 tool_calls），ui 契约稳定。"""
        response = asyncio.run(backend.get_llm_response("你好"))
        assert response.content == "好的"
        assert response.tool_calls == []


class FakeJudge:
    """判定器替身：记录收到的文本，可编排判定结果。"""

    def __init__(self, result=True):
        self.result = result
        self.calls = []

    async def should_reply(self, user_text):
        self.calls.append(user_text)
        return self.result


class TestJudgeChannel:
    """openai 后端判定通道（2026-08-14）：judge 上下文路由到独立判定器。"""

    def test_judge_context_routes_to_judge(self, tmp_path):
        """判定上下文（首条 system = JUDGE_SYSTEM_PROMPT）走独立判定器，不进主 client。"""
        judge = FakeJudge()
        backend = OpenAIBackend(client=_StubClient(), db_path=tmp_path / "pet.db", judge=judge)
        context = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "用户的消息是：你在吗"},
        ]
        response = asyncio.run(backend.get_response_with_context(context))
        assert response.content == "true"
        assert judge.calls == ["你在吗"]

    def test_memory_context_bypasses_judge(self, tmp_path):
        """记忆上下文（摘要提示词）不匹配判定通道，仍走主 client。"""
        judge = FakeJudge()
        stub = _StubClient()
        backend = OpenAIBackend(client=stub, db_path=tmp_path / "pet.db", judge=judge)
        context = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "已有摘要：…"},
        ]
        response = asyncio.run(backend.get_response_with_context(context))
        assert response.content == "好的"
        assert judge.calls == []

    def test_judge_false_content(self, tmp_path):
        judge = FakeJudge(result=False)
        backend = OpenAIBackend(client=_StubClient(), db_path=tmp_path / "pet.db", judge=judge)
        context = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": "用户的消息是：嗯嗯"},
        ]
        response = asyncio.run(backend.get_response_with_context(context))
        assert response.content == "false"


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

    def test_extract_adds_fact_and_advances_cursor(self, backend):
        backend.client.reply = '{"operations": [{"op": "add", "content": "用户在 UESTC 读书"}]}'

        async def run():
            await backend.get_llm_response("我在 UESTC 读书")
            await backend.maybe_extract_facts()
            # 没有新轮次时不重复抽取、不重复落库
            await backend.maybe_extract_facts()

        asyncio.run(run())
        assert [c for _, c in backend.memory.get_facts()] == ["用户在 UESTC 读书"]
        assert backend.memory.get_meta("last_extracted_turn") == str(backend.turn_id)

    def test_extract_bad_output_discarded(self, backend):
        backend.client.reply = "胡说八道"

        async def run():
            await backend.get_llm_response("你好")
            await backend.maybe_extract_facts()

        asyncio.run(run())
        assert backend.memory.get_facts() == []
        # 游标照常推进，不会反复重试同一批
        assert backend.memory.get_meta("last_extracted_turn") == str(backend.turn_id)

    def test_extract_covers_memorized_turns(self, backend):
        backend.memorize("我下周三要交实验报告")
        backend.client.reply = '{"operations": [{"op": "add", "content": "用户下周三要交实验报告"}]}'

        async def run():
            await backend.get_llm_response("糯糯你在吗")
            await backend.maybe_extract_facts()

        asyncio.run(run())
        assert [c for _, c in backend.memory.get_facts()] == ["用户下周三要交实验报告"]

    def test_apply_fact_ops_validation(self, backend):
        backend.memory.add_fact("旧事实")
        fid = backend.memory.get_facts()[0][0]
        backend._apply_fact_ops(
            [
                {"op": "update", "id": fid, "content": "新事实"},
                {"op": "update", "id": 999, "content": "不存在的 id"},
                {"op": "add", "content": "   "},
                {"op": "delete", "id": fid},
                {"op": "unknown"},
            ]
        )
        assert backend.memory.get_facts() == []

    def test_facts_injected_into_messages(self, backend):
        backend.memory.add_fact("用户叫小明")
        messages = backend._build_messages()
        assert any("用户叫小明" in m["content"] for m in messages[:2])

    def test_merge_facts_over_limit(self, backend):
        for i in range(MAX_FACTS + 1):
            backend.memory.add_fact(f"事实{i}")
        backend.client.reply = '["合并事实1", "合并事实2"]'
        asyncio.run(backend._maybe_merge_facts())
        assert [c for _, c in backend.memory.get_facts()] == ["合并事实1", "合并事实2"]

    def test_merge_bad_output_keeps_original(self, backend):
        for i in range(MAX_FACTS + 1):
            backend.memory.add_fact(f"事实{i}")
        backend.client.reply = "不是json"
        asyncio.run(backend._maybe_merge_facts())
        assert len(backend.memory.get_facts()) == MAX_FACTS + 1
