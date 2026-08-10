"""回复后端工厂：按 config.toml 的 BACKEND 键装配实现（astrbot / openai）。

新增后端：在 backends/ 下实现 ReplyBackend 子类，这里加一个分支即可。
"""

from openai import AsyncOpenAI

from .astrbot import AstrBotBackend
from .base import BackendResponse, ReplyBackend, ToolCall
from .judger import LLMJudge
from .openai import OpenAIBackend


def create_judge(config):
    """构建 LLM 决策器（should_reply 判定，astrbot 后端用）。

    复用 openai 后端的 API_KEY/BASE_URL；模型用 JUDGE_MODEL（不填则默认模型）。
    API_KEY 缺失时返回 None（后端会回退"默认回复"）。
    """
    api_key = config.get("API_KEY")
    base_url = config.get("BASE_URL")
    if not api_key or not base_url:
        print("判定器未配置：config.toml 缺少 API_KEY/BASE_URL（openai 段），判定回退为默认回复")
        return None
    model = config.get("JUDGE_MODEL", "gemini-2.5-pro")
    return LLMJudge(client=AsyncOpenAI(api_key=api_key, base_url=base_url), model=model)


def create_backend(config):
    """根据配置创建回复后端。config 为 tomllib 解析出的 dict。"""
    name = config.get("BACKEND", "astrbot")
    if name == "astrbot":
        return AstrBotBackend(
            judge=create_judge(config),
            url=config.get("ASTRBOT_WS_URL", "ws://127.0.0.1:20081/api"),
            token=config.get("ASTRBOT_WS_TOKEN", ""),
            user_id=config.get("ASTRBOT_USER_ID", 1063310598),
            self_id=config.get("ASTRBOT_SELF_ID", 10001),
            timeout=config.get("ASTRBOT_TIMEOUT", 90.0),
            settle=config.get("ASTRBOT_SETTLE", 2.0),
            screen_idle_interval=config.get("SCREEN_IDLE_INTERVAL", 60),
            screen_active_interval=config.get("SCREEN_ACTIVE_INTERVAL", 20),
            screen_change_threshold=config.get("SCREEN_CHANGE_THRESHOLD", 0.05),
            screen_cooldown=config.get("SCREEN_COOLDOWN", 180),
            screen_user_silence=config.get("SCREEN_USER_SILENCE", 120),
        )
    if name == "openai":
        return OpenAIBackend()
    raise ValueError(f"未知 BACKEND: {name!r}（可选：astrbot / openai）")


__all__ = ["BackendResponse", "ReplyBackend", "ToolCall", "create_backend"]
