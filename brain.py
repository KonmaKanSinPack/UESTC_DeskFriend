"""brain 门面：ui.py 与回复后端之间的薄适配层。

ui.py 依赖的契约（保持不变）：
- Brain 五个方法：get_llm_response / get_response_with_context / memorize / maybe_compress / maybe_extract_facts
- 工具函数：pack_msg / parse_tool_args
具体"大脑"由 backends/ 按 config.toml 的 BACKEND 键选择（astrbot / openai），
统一返回 BackendResponse（content + tool_calls），ui 不感知后端差异。
"""

import json

import tomllib

from backends import BackendResponse, create_backend


def pack_msg(role, type, content, tool_call=None):
    if type == "text":
        return {"role": role, "content": content}
    elif type == "image_url":
        return {"role": role, "content": [{"type": "image_url", "image_url": {"url": content}}]}
    elif type == "tool":
        # tool_call 是统一 ToolCall 对象（id / name / arguments）
        return {"role": "tool", "tool_call_id": tool_call.id, "name": tool_call.name, "content": content}


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
    """门面：按配置装配后端，原样转发五个接口；reply_sink 转发给支持主动冒泡的后端。"""

    def __init__(self, backend=None):
        """backend 可注入（测试用）；生产路径从 config.toml 经工厂创建。"""
        if backend is None:
            with open("config.toml", "rb") as f:
                config = tomllib.load(f)
            backend = create_backend(config)
        self.backend = backend

    async def stop(self):
        """停止后端后台任务（连接循环/观察循环），进程退出前调用。"""
        stop = getattr(self.backend, "stop", None)
        if stop is not None:
            await stop()

    def set_interruption(self, prefix: str):
        """记录 TTS 朗读被打断的位置（已播放文本前缀）。

        由 ui 在用户新消息打断朗读时调用；astrbot 后端下次对话时把它
        注入消息上下文（让桃桃知道"说到哪被打断了"），用后清除。
        """
        if prefix and hasattr(self.backend, "interruption"):
            self.backend.interruption = prefix

    @property
    def reply_sink(self):
        """主动冒泡回调：后端（astrbot 的屏幕感知）有值得说的话时调用，ui 挂 show_bubble。"""
        return getattr(self.backend, "reply_sink", None)

    @reply_sink.setter
    def reply_sink(self, fn):
        if hasattr(self.backend, "reply_sink"):
            self.backend.reply_sink = fn

    # ---- 五个接口：原样转发 ----

    async def get_llm_response(self, message, model=None) -> BackendResponse:
        """对话主通道：发送消息并返回回复。"""
        return await self.backend.get_llm_response(message, model)

    async def get_response_with_context(self, context, model=None, use_tools=False) -> BackendResponse:
        """should_reply 判定：content 为 "true"/"false"。"""
        return await self.backend.get_response_with_context(context, model, use_tools)

    def memorize(self, message):
        """无需回复的背景谈话：只记入记忆（若后端有）。"""
        self.backend.memorize(message)

    async def maybe_compress(self):
        """记忆压缩：超限时把最老轮次并入摘要，失败不影响对话。"""
        await self.backend.maybe_compress()

    async def maybe_extract_facts(self):
        """事实抽取：从近期对话提炼长期事实，失败不影响对话。"""
        await self.backend.maybe_extract_facts()
