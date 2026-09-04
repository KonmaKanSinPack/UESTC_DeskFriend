"""配置装配层工具：加载 config/ 下的分层配置并合并。

结构（2026-09-04 重构，方案 A 经用户确认）：
    config/common.toml     BACKEND 选择器 + 两后端共用键（ASR_*/TTS_*/SPRITE/JUDGE_*/MEMORY_*）
    config/openai.toml     openai 后端专属（API_KEY/BASE_URL/OPENAI_MODEL/SYSTEM_PROMPT）
    config/astrbot.toml    astrbot 后端专属（ASTRBOT_*/SCREEN_*）

合并规则：common 读出 BACKEND → 加载对应后端文件 → 后端键覆盖同名共享键
（如 openai.toml 的真实 API_KEY 覆盖 common 里判定器的本地兜底密钥）。

load_config 从 skin.py 迁来（历史遗留）：配置加载是装配层的职责，不是皮的——
skin 只消费 SPRITE 一个键，brain/spine/mouth 经它拿各自的段。
"""

from pathlib import Path

import tomllib

PROJECT_DIR = Path(__file__).parent
CONFIG_DIR = PROJECT_DIR / "config"


class ConfigError(RuntimeError):
    """配置缺失/不合法——消息里带可操作的修复指引，而不是让上层猜。"""


def load_config(config_dir=None) -> dict:
    """读 common.toml + 按 BACKEND 键合并对应后端文件，返回合并后的配置 dict。

    config_dir 可注入（测试用）；默认项目根 config/ 目录。
    """
    d = Path(config_dir) if config_dir else CONFIG_DIR

    common_path = d / "common.toml"
    if not common_path.exists():
        raise ConfigError(
            f"缺少 {common_path}——请从 config/common.example.toml 复制创建，"
            "填 BACKEND 与共享配置（判定器/TTS/ASR/贴图/记忆阈值）"
        )
    with open(common_path, "rb") as f:
        merged = tomllib.load(f)

    # 默认 astrbot 与 create_backend 的缺省一致（同一约定两处不改会漂移，此处显式）
    backend = merged.get("BACKEND", "astrbot")
    backend_path = d / f"{backend}.toml"
    if not backend_path.exists():
        raise ConfigError(
            f"BACKEND = {backend!r} 但缺少 {backend_path}——请从 config/{backend}.example.toml 复制创建并填入密钥"
        )
    with open(backend_path, "rb") as f:
        merged.update(tomllib.load(f))  # 后端文件覆盖同名共享键
    return merged
