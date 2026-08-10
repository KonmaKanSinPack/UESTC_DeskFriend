"""LLMJudge（判定器）测试：提示词解析 true/false、判定失败回退。"""

import asyncio
from types import SimpleNamespace

from backends.judger import JUDGE_SYSTEM_PROMPT, LLMJudge


class _StubClient:
    """假 LLM：固定回复 self.reply。"""

    def __init__(self, reply="true"):
        self.reply = reply
        self.chat = SimpleNamespace(completions=self)
        self.last_messages = None

    async def create(self, model, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=self.reply, tool_calls=None))]
        )


def test_judge_true():
    judge = LLMJudge(client=_StubClient("true"), model="m1")
    assert asyncio.run(judge.should_reply("糯糯你在吗")) is True


def test_judge_false():
    judge = LLMJudge(client=_StubClient("false"), model="m1")
    assert asyncio.run(judge.should_reply("嗯嗯")) is False


def test_judge_case_and_whitespace_tolerant():
    for raw in ("True", " TRUE ", "true\n"):
        judge = LLMJudge(client=_StubClient(raw), model="m1")
        assert asyncio.run(judge.should_reply("在吗")) is True


def test_judge_prompt_contains_few_shots():
    client = _StubClient("true")
    judge = LLMJudge(client=client, model="m1")
    asyncio.run(judge.should_reply("看看我的屏幕"))
    msgs = client.last_messages
    assert msgs[0]["role"] == "system"
    assert JUDGE_SYSTEM_PROMPT in msgs[0]["content"]
    assert "用户的消息是：看看我的屏幕" in msgs[1]["content"]


def test_judge_failure_falls_back_to_true():
    class BoomClient:
        async def create(self, model, messages, **kwargs):
            raise RuntimeError("网络挂了")

    judge = LLMJudge(client=BoomClient(), model="m1")
    assert asyncio.run(judge.should_reply("在吗")) is True


def test_think_block_stripped_before_matching():
    """推理模型（如 MiniMax-M3）输出 <think> 思考块 + 结论，剥块后再匹配。"""
    judge = LLMJudge(client=_StubClient("<think>The user is calling the pet</think>\n\ntrue"), model="m1")
    assert asyncio.run(judge.should_reply("糯糯在吗")) is True


def test_no_verdict_returns_false():
    """模型没输出 true/false 词（如被截断）→ 保守 false，不误判。"""
    judge = LLMJudge(client=_StubClient("<think>thinking...</think>"), model="m1")
    assert asyncio.run(judge.should_reply("在吗")) is False


def test_max_tokens_room_for_reasoning():
    """max_tokens 要给思考块留空间（曾用 8 被截断导致判定全 false）。"""
    client = _StubClient("true")
    judge = LLMJudge(client=client, model="m1")
    asyncio.run(judge.should_reply("在吗"))
    assert client.last_kwargs["max_tokens"] >= 100
