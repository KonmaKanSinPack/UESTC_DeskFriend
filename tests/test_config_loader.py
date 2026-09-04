"""config_loader 单测：三文件分层加载/合并/缺文件指引。全部用 tmp_path，不碰真配置。"""

import pytest

from config_loader import ConfigError, load_config


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


COMMON = """
BACKEND = "openai"
TTS_BACKEND = "siliconflow"
API_KEY = "lm-studio"
"""

OPENAI = """
API_KEY = "sk-real"
OPENAI_MODEL = "gemini-2.5-pro"
"""

ASTRBOT = """
ASTRBOT_WS_URL = "ws://127.0.0.1:20081/api"
"""


@pytest.fixture()
def cfg_dir(tmp_path):
    _write(tmp_path / "common.toml", COMMON)
    _write(tmp_path / "openai.toml", OPENAI)
    _write(tmp_path / "astrbot.toml", ASTRBOT)
    return tmp_path


def test_common_merged_with_selected_backend(cfg_dir):
    """common 读出 BACKEND → 合并对应后端文件，共享键齐全。"""
    cfg = load_config(cfg_dir)
    assert cfg["BACKEND"] == "openai"
    assert cfg["TTS_BACKEND"] == "siliconflow"  # 共享键来自 common
    assert cfg["OPENAI_MODEL"] == "gemini-2.5-pro"  # 后端专属键来自 openai.toml


def test_backend_keys_override_common(cfg_dir):
    """后端文件的同名键覆盖共享键：openai 真实 API_KEY 盖掉判定器兜底密钥。"""
    cfg = load_config(cfg_dir)
    assert cfg["API_KEY"] == "sk-real"


def test_switch_backend_via_common(cfg_dir):
    _write(cfg_dir / "common.toml", COMMON.replace('"openai"', '"astrbot"'))
    cfg = load_config(cfg_dir)
    assert cfg["ASTRBOT_WS_URL"] == "ws://127.0.0.1:20081/api"
    assert "OPENAI_MODEL" not in cfg  # openai 专属键不泄入 astrbot 模式
    assert cfg["API_KEY"] == "lm-studio"  # 未被覆盖：判定器兜底密钥生效


def test_missing_common_tells_how_to_fix(tmp_path):
    with pytest.raises(ConfigError, match="common.example.toml"):
        load_config(tmp_path)


def test_missing_backend_file_tells_how_to_fix(tmp_path):
    _write(tmp_path / "common.toml", COMMON)  # 只建 common，缺 openai.toml
    with pytest.raises(ConfigError, match=r"openai\.example\.toml"):
        load_config(tmp_path)
