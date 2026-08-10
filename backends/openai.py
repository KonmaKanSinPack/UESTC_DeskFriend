"""openai 回复后端：直连 OpenAI 兼容接口 + 自建记忆（原 brain.py 整体迁移）。

行为与旧实现一致：tool calling 循环（look_at_screen）、SQLite 持久化上下文、
滚动摘要压缩、事实型长期记忆。仅将返回值归一为 BackendResponse，ui 无感。
"""

import json

import tomllib
from openai import AsyncOpenAI

from memory import KEEP_RECENT_TURNS, MAX_CONTEXT_TURNS, MemoryStore, render_messages

from .base import BackendResponse, ReplyBackend, ToolCall

# 默认人设：糯糯（Q版弗洛洛毛绒玩偶）。可在 config.toml 用 SYSTEM_PROMPT 覆盖。
DEFAULT_SYSTEM_PROMPT = (
    "你是糯糯，一只Q版弗洛洛毛绒玩偶形态的桌面AI伙伴，住在用户的电脑桌面上。"
    "用中文回复，语气软萌、自然、像朋友闲聊，回复要简短（一两句话），"
    "不要用 markdown、列表或标题，因为回复会显示在一个小气泡里。"
)

# 摘要压缩提示词：已有摘要 + 新对话 → 输出更新后的完整摘要（滚动更新）
SUMMARY_SYSTEM_PROMPT = (
    "你是记忆整理助手。把桌宠与用户的对话压缩成一段持续更新的摘要，"
    "保留：用户透露的个人信息与偏好、重要约定、未完结的话题、关键事件。"
    "用中文，200 字以内，直接输出摘要正文，不要任何前缀或解释。"
)

# 事实抽取：每次回复后处理自上次抽取以来的全部轮次（含背景谈话）
FACT_EXTRACTION_MAX_TURNS = 20
# facts 条数上限，超过后由 LLM 合并压缩到 MERGED_FACTS_TARGET 条
MAX_FACTS = 50
MERGED_FACTS_TARGET = 30

FACT_EXTRACTION_SYSTEM_PROMPT = (
    "你是桌宠的记忆提取助手。从对话中抽取值得长期记住的关于用户的事实"
    "（身份、学校、偏好、计划、重要事件、与别人提到的约定等，包括标注为背景谈话的内容）。"
    "规则：只记有长期价值的事实，不记寒暄和一次性内容；"
    "已存在的事实不要重复添加；内容变化时用 update 更新，失效时用 delete 删除；"
    "每条事实是一句简短中文陈述。"
    '输出纯 JSON：{"operations": [{"op": "add", "content": "..."}, '
    '{"op": "update", "id": 1, "content": "..."}, {"op": "delete", "id": 2}]}，'
    '没有要操作的内容时输出 {"operations": []}。不要输出任何其他文字。'
)

FACT_MERGE_SYSTEM_PROMPT = (
    "你是记忆整理助手。把以下关于用户的事实列表合并压缩，"
    "去掉重复、合并相近条目，保留全部有效信息。"
    "输出纯 JSON 字符串数组，不要输出任何其他文字。"
)


def parse_fact_ops(raw):
    """解析事实抽取 LLM 输出的 JSON 操作列表，任何异常都兜底为空列表。"""
    try:
        text = raw.strip()
        if text.startswith("```"):
            # 去掉 markdown 代码围栏
            text = text.strip("`").removeprefix("json").strip()
        data = json.loads(text)
        ops = data.get("operations") if isinstance(data, dict) else None
        return [op for op in ops if isinstance(op, dict)] if isinstance(ops, list) else []
    except Exception:
        return []


class OpenAIBackend(ReplyBackend):
    name = "openai"

    def __init__(self, client=None, db_path=None, system_prompt=None):
        """client / db_path / system_prompt 可注入，仅供测试；生产路径从 config.toml 构建。"""
        if client is None:
            with open("config.toml", "rb") as f:
                config = tomllib.load(f)
            client = AsyncOpenAI(api_key=config["API_KEY"], base_url=config["BASE_URL"])
            system_prompt = config.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.client = client

        self.cur_model = "gemini-2.5-pro"
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "look_at_screen",
                    "description": "look at the current screen",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        # 记忆：启动时从 pet.db 恢复滚动摘要 + 未压缩的历史消息
        self.memory = MemoryStore(db_path) if db_path else MemoryStore()
        self.summary = self.memory.get_summary()
        self.context = self.memory.load_unsummarized()
        self.turn_id = self.memory.max_turn_id()

    # ---------- 内部工具 ----------

    def _store(self, msg):
        """消息同时进内存 context 和 pet.db，两者保持一一对应、顺序一致。"""
        self.context.append(msg)
        self.memory.add_message(self.turn_id, msg)

    def _build_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        facts = self.memory.get_facts()
        if facts:
            lines = "\n".join(f"{i}. {content}" for i, (_, content) in enumerate(facts, 1))
            messages.append({"role": "system", "content": f"以下是你记住的关于用户的事情：\n{lines}"})
        if self.summary:
            messages.append({"role": "system", "content": f"以下是你和用户此前对话的摘要：\n{self.summary}"})
        messages += self.context
        return messages

    def _to_backend_response(self, response):
        """SDK 响应 → BackendResponse；assistant 消息同时落库（tool_calls 必须保留）。"""
        resp_msg = response.choices[0].message
        response_msg = {"role": resp_msg.role, "content": resp_msg.content}
        tool_calls = []
        if resp_msg.tool_calls:
            # tool_calls 必须保留，否则后续 tool 角色消息找不到对应调用，API 返回 400
            response_msg["tool_calls"] = [tc.model_dump() for tc in resp_msg.tool_calls]
            tool_calls = [
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments) for tc in resp_msg.tool_calls
            ]
        self._store(response_msg)
        return BackendResponse(content=resp_msg.content or "", tool_calls=tool_calls)

    # ---------- 对话 ----------

    async def get_llm_response(self, message, model=None):
        """
        message 和回复都会存入 context 并落库。
        message 可以是 str（普通用户消息，开启新一轮），也可以是已打包好的 dict
        （如 tool 结果消息），或 dict 列表（需要按序追加多条消息时）。
        """
        if isinstance(message, list):
            msgs = message
        elif isinstance(message, dict):
            msgs = [message]
        else:
            self.turn_id = self.memory.next_turn_id()
            msgs = [{"role": "user", "content": message}]
        for m in msgs:
            self._store(m)
        if model is None:
            model = self.cur_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=self._build_messages(),
            # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
            extra_body={"reasoning_split": True},
            tools=self.tools,
            tool_choice="auto",
        )
        return self._to_backend_response(response)

    async def get_response_with_context(self, context, model=None, use_tools=False):
        if model is None:
            model = self.cur_model
        kwargs = {}
        if use_tools:
            # 判定类调用（如 should_reply）不能带 tools：
            # 模型可能直接发起工具调用而不输出文本，导致判定落空
            kwargs = {"tools": self.tools, "tool_choice": "auto"}
        response = await self.client.chat.completions.create(
            model=model,
            messages=context,
            # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
            extra_body={"reasoning_split": True},
            **kwargs,
        )
        return BackendResponse(content=response.choices[0].message.content or "")

    # ---------- 记忆 ----------

    def memorize(self, message):
        """判断为无需回复的听觉消息：只落记忆、不生成回复。

        作为带标注的独立新轮次记录，模型后续能在上下文中看到这些背景谈话
        （事实抽取也会覆盖它们），但知道不需要回应。
        """
        self.turn_id = self.memory.next_turn_id()
        self._store({"role": "user", "content": f"[背景谈话，无需回应] {message}"})

    async def maybe_compress(self):
        """未压缩轮次超限时，把最老轮次增量并入滚动摘要。自身吞掉异常，绝不影响对话。"""
        try:
            if self.memory.unsummarized_turn_count() <= MAX_CONTEXT_TURNS:
                return
            turn_ids, old_msgs = self.memory.turns_to_summarize(KEEP_RECENT_TURNS)
            if not turn_ids:
                return
            summary_context = [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"已有摘要：\n{self.summary or '（无）'}\n\n"
                        f"需要并入的新对话：\n{render_messages(old_msgs)}\n\n"
                        "请输出更新后的完整摘要。"
                    ),
                },
            ]
            response = await self.get_response_with_context(summary_context)
            new_summary = response.content.strip()
            if not new_summary:
                return
            self.memory.set_summary(new_summary)
            self.memory.mark_summarized(turn_ids)
            self.summary = new_summary
            # 内存 context 与未压缩消息一一对应，最前面的 len(old_msgs) 条即被压缩的部分
            del self.context[: len(old_msgs)]
            print(f"记忆压缩：{len(turn_ids)} 个旧轮次已并入摘要")
        except Exception as e:
            print(f"记忆压缩失败：{e}")

    async def maybe_extract_facts(self):
        """处理自上次抽取以来的全部轮次（含背景谈话），让 LLM 输出事实操作并落库。

        只在回复完成后调用：背景消息攒着，下次用户搭话时一并批量抽取，
        避免每条背景消息都花一次 LLM 调用。自身吞掉异常，绝不影响对话。
        """
        try:
            last = int(self.memory.get_meta("last_extracted_turn", "0"))
            turn_ids, msgs = self.memory.get_turns_since(last, FACT_EXTRACTION_MAX_TURNS)
            if not turn_ids:
                return
            facts = self.memory.get_facts()
            facts_text = "\n".join(f"{fid}. {content}" for fid, content in facts) or "（无）"
            extract_context = [
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"已有事实：\n{facts_text}\n\n最近对话：\n{render_messages(msgs)}\n\n请输出操作 JSON。",
                },
            ]
            response = await self.get_response_with_context(extract_context)
            # 已成功拿到响应即推进游标：坏输出丢弃本批，避免反复重试同一批轮次
            self.memory.set_meta("last_extracted_turn", str(max(turn_ids)))
            ops = parse_fact_ops(response.content)
            if ops:
                self._apply_fact_ops(ops)
                print(f"事实抽取：应用 {len(ops)} 条操作")
            await self._maybe_merge_facts()
        except Exception as e:
            print(f"事实抽取失败：{e}")

    def _apply_fact_ops(self, ops):
        existing_ids = {fid for fid, _ in self.memory.get_facts()}
        for op in ops:
            action = op.get("op")
            content = op.get("content")
            fact_id = op.get("id")
            if action == "add" and isinstance(content, str) and content.strip():
                self.memory.add_fact(content.strip())
            elif action == "update" and fact_id in existing_ids and isinstance(content, str) and content.strip():
                self.memory.update_fact(fact_id, content.strip())
            elif action == "delete" and fact_id in existing_ids:
                self.memory.delete_fact(fact_id)

    async def _maybe_merge_facts(self):
        """facts 超上限时让 LLM 合并压缩；坏输出不落库，原表不动。"""
        facts = self.memory.get_facts()
        if len(facts) <= MAX_FACTS:
            return
        merge_context = [
            {"role": "system", "content": FACT_MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请把以下 {len(facts)} 条事实合并到不超过 {MERGED_FACTS_TARGET} 条：\n"
                    + "\n".join(content for _, content in facts)
                ),
            },
        ]
        response = await self.get_response_with_context(merge_context)
        try:
            merged = json.loads(response.content.strip().strip("`").removeprefix("json"))
        except Exception:
            return
        if isinstance(merged, list):
            contents = [c.strip() for c in merged if isinstance(c, str) and c.strip()]
            if contents:
                self.memory.replace_facts(contents)
                print(f"事实合并：{len(facts)} 条压缩为 {len(contents)} 条")
