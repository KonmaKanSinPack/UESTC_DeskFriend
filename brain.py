import json
from collections import deque

import tomllib
from openai import AsyncOpenAI


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


class Brain:
    def __init__(self):
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        self.context = deque(maxlen=5)
        self.cur_model = "gemini-2.5-pro"
        self.client = AsyncOpenAI(api_key=config["API_KEY"], base_url=config["BASE_URL"])
        self.system_prompt = config.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
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

    async def get_llm_response(self, message, model=None):
        """
        message和response_msg都会直接存入context。
        message 可以是 str（普通用户消息），也可以是已打包好的 dict（如 tool 结果消息），
        或 dict 列表（需要按序追加多条消息时）。
        """
        if isinstance(message, list):
            self.context.extend(message)
        elif isinstance(message, dict):
            self.context.append(message)
        else:
            self.context.append({"role": "user", "content": message})
        if model is None:
            model = self.cur_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.context,
            ],
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
        self.context.append(response_msg)
        return response

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
