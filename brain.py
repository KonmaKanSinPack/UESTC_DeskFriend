import json

import tomllib
from openai import AsyncOpenAI

from memory import KEEP_RECENT_TURNS, MAX_CONTEXT_TURNS, MemoryStore, render_messages


def pack_msg(role, type, content, tool_call=None):
    if type == "text":
        return {"role": role, "content": content}
    elif type == "image_url":
        return {"role": role, "content": [{"type": "image_url", "image_url": {"url": content}}]}
    elif type == "tool":
        return {"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.function.name, "content": content}


def parse_tool_args(args_str):
    """解析大模型发来的工具参数 JSON，任何异常都兜底为空字典。"""
    try:
        # 尝试解析
        args_dict = json.loads(args_str) if args_str else {}
        # 如果解析出来是 None (比如遇到了 "null")，或者不是字典，强制兜底为空字典
        if not isinstance(args_dict, dict):
            args_dict = {}
    except Exception:
        # 万一大模型抽风发来一段根本无法解析的乱码，也用空字典兜底
        args_dict = {}
    return args_dict


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


class Brain:
    def __init__(self, client=None, db_path=None):
        """client / db_path 可注入，仅供测试；生产路径从 config.toml 与默认 pet.db 构建。"""
        if client is None:
            with open("config.toml", "rb") as f:
                config = tomllib.load(f)
            client = AsyncOpenAI(api_key=config["API_KEY"], base_url=config["BASE_URL"])
            self.system_prompt = config.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
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

    def _store(self, msg):
        """消息同时进内存 context 和 pet.db，两者保持一一对应、顺序一致。"""
        self.context.append(msg)
        self.memory.add_message(self.turn_id, msg)

    def _build_messages(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({"role": "system", "content": f"以下是你和用户此前对话的摘要：\n{self.summary}"})
        messages += self.context
        return messages

    def memorize(self, message):
        """判断为无需回复的听觉消息：只落记忆、不生成回复。

        作为带标注的独立新轮次记录，模型后续能在上下文中看到这些背景谈话
        （Phase 6 的事实抽取也会覆盖它们），但知道不需要回应。
        """
        self.turn_id = self.memory.next_turn_id()
        self._store({"role": "user", "content": f"[背景谈话，无需回应] {message}"})

    async def get_llm_response(self, message, model=None):
        """
        message和response_msg都会存入 context 并落库。
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
        resp_msg = response.choices[0].message
        response_msg = {"role": resp_msg.role, "content": resp_msg.content}
        if resp_msg.tool_calls:
            # tool_calls 必须保留，否则后续 tool 角色消息找不到对应调用，API 返回 400
            response_msg["tool_calls"] = [tc.model_dump() for tc in resp_msg.tool_calls]
        self._store(response_msg)
        return response

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
            new_summary = (response.choices[0].message.content or "").strip()
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
        # resp_message = response.choices[0].message
        return response
