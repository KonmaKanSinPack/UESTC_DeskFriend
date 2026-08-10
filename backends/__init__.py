"""回复后端工厂：按 config.toml 的 BACKEND 键装配实现（astrbot / openai）。

新增后端：在 backends/ 下实现 ReplyBackend 子类，这里加一个分支即可。
"""

from .astrbot import AstrBotBackend
from .base import BackendResponse, ReplyBackend, ToolCall
from .openai import OpenAIBackend


def create_backend(config):
    """根据配置创建回复后端。config 为 tomllib 解析出的 dict。"""
    name = config.get("BACKEND", "astrbot")
    if name == "astrbot":
        return AstrBotBackend(
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
