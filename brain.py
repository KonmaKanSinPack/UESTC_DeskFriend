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


class Brain:
    def __init__(self):
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)

        self.context = deque(maxlen=5)
        self.cur_model = "gemini-2.5-pro"
        self.client = AsyncOpenAI(api_key=config["API_KEY"], base_url=config["BASE_URL"])
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
        """
        self.context.append({"role": "user", "content": message})
        if model is None:
            model = self.cur_model

        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant.中文回复"},
                *self.context,
            ],
            # 设置 reasoning_split=True 将思考内容分离到 reasoning_details 字段
            extra_body={"reasoning_split": True},
            tools=self.tools,
            tool_choice="auto",
        )
        response_msg = {"role": response.choices[0].message.role, "content": response.choices[0].message.content}
        self.context.append(response_msg)
        # resp_message = response.choices[0].message
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
