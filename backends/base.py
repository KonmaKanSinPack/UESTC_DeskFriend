"""回复后端统一契约：brain 门面与 ui 只认这里的类型，不依赖具体后端的 SDK 类型。

后端 = 桌宠的"大脑"实现，两种：
- astrbot：经 OneBot 11 伪装通道接入 AstrBot 的桃桃（记忆/人格由 AstrBot 接管）
- openai：直连 OpenAI 兼容接口 + 自建记忆（memory.py）

ui.py 的依赖面：BackendResponse.content（气泡文本）、ToolCall.id/name/arguments（工具循环）。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """归一化工具调用：openai 后端把 SDK 对象转成这个，astrbot 后端不产生工具调用。"""

    id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class BackendResponse:
    """统一回复对象：两种后端都返回它，ui 无感。"""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    answered: bool = True  # False=桥超时/断线兜底，非真实回复；主控不朗读（默认 True → openai 后端零改动）
    speak: bool = True  # False=只显示不朗读（主动观察空闲期：仅活跃期朗读）；与 answered 语义独立（2026-09-03）


class ReplyBackend(ABC):
    """回复后端抽象。实现者必须保证三件事：
    1. 五个方法语义与旧 brain 一致（ui.py 的调用方式不变）；
    2. get_llm_response 的 message 入参支持 str / dict / list[dict] 三种形态；
    3. 记忆相关接口（memorize/maybe_compress/maybe_extract_facts）要么真实现（openai），
       要么显式空操作（astrbot，记忆由 AstrBot 接管）。
    """

    name: str = "abstract"

    @abstractmethod
    async def get_llm_response(self, message, model=None) -> BackendResponse:
        """对话主通道：发送消息并返回回复。message 为 str 时开启新一轮。"""

    @abstractmethod
    async def get_response_with_context(self, context, model=None, use_tools=False) -> BackendResponse:
        """should_reply 判定通道：content 为 "true"/"false"。

        判定类调用不应携带 tools（模型可能用工具调用代替回答导致判定落空）。
        """

    @abstractmethod
    def memorize(self, message):
        """无需回复的背景谈话：只记入记忆（若后端有），不生成回复。注意是同步方法（ui 未 await）。"""

    @abstractmethod
    async def maybe_compress(self):
        """记忆压缩（上下文轮次超限时）；失败不得影响对话。"""

    @abstractmethod
    async def maybe_extract_facts(self):
        """事实抽取（从近期对话提炼长期事实）；失败不得影响对话。"""
