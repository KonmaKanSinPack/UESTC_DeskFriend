# UESTC_DeskFriend

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)

桌面 AI 伙伴（桌宠），形象为糯糯（Q版弗洛洛）。常驻桌面一隅，能听、能看、能聊：

- **听觉**：麦克风常驻监听，Silero VAD（ONNX，本地模型）检测说话，faster-whisper 中文语音转文字
- **视觉**：截屏注入多模态大模型上下文，说一句"看看我的屏幕"它就能描述你在干什么
- **对话**：OpenAI 兼容接口（默认 `gemini-2.5-pro`），支持 tool calling，气泡显示回复
- **交互**：语音 / 单击输入文字 / 双击"触碰"彩蛋 / 拖动摆放，带呼吸、思考、说话的状态动画

## 运行环境

- **Linux**（开发环境为 Ubuntu GNOME **Wayland**；X11 亦可，功能降级见下）
- **Windows 10/11**
- Python 3.12（由 uv 管理）
- 可选 NVIDIA GPU（Whisper 自动检测，无显卡或 CUDA 依赖不全时回退 CPU int8）

### Linux 系统依赖

```bash
sudo apt install -y libportaudio2 libxcb-xinerama0
# 静默截屏（Wayland）需要：
sudo apt install -y python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-pipewire
```

- `libportaudio2`：音频采集（sounddevice）
- `libxcb-xinerama0`：PyQt5 xcb 插件运行库
- gi/GStreamer 三件套：Wayland 下 PipeWire 静默截屏。缺失时回退 `gnome-screenshot`（有闪光灯效）

### Windows

- 无需额外系统依赖（`sounddevice` 的 Windows wheel 已内置 PortAudio）
- 截屏使用 Pillow `ImageGrab`，支持多显示器，无需授权
- 首次运行如弹出麦克风权限请求，请允许
- 有 NVIDIA 显卡但未安装 CUDA/cuBLAS/cuDNN 时，Whisper 会自动回退 CPU，不影响启动

### 安装

Linux（venv 必须带 `--system-site-packages`，否则用不上系统的 gi，静默截屏失效）：

```bash
uv venv --system-site-packages --python 3.12
uv sync
```

Windows（PowerShell）：

```powershell
uv sync --python 3.12
```

### 配置

```bash
cp config.example.toml config.toml
# 编辑 config.toml，填入 API_KEY 和 BASE_URL
# 可选：SYSTEM_PROMPT 自定义系统提示词（人设），不填用内置糯糯人设
# 可选：SPRITE 自定义贴图路径，支持 GIF 动图（默认 assets/nuonuo.png）
```

Windows 下把 `cp` 换成 `Copy-Item config.example.toml config.toml`。

### 模型

- VAD 模型已内置（`assets/silero_vad.onnx`），无需下载
- faster-whisper `small` 首次使用需下载（约 460MB）。之后启动全部离线
- 网络不佳时用镜像下载：

```bash
# Linux / macOS
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

```powershell
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

### 运行

```bash
uv run python main.py
```

Linux 首次运行会弹一次 GNOME 屏幕共享授权框（PipeWire 截屏），点允许后永久静默，授权 token 存于 `pw_restore_token`（删除该文件可重新授权）。Windows 无此步骤。

`Ctrl+C` 退出；Windows 也可以直接从任务栏关闭窗口。若不想保留终端窗口，可用 `.venv\Scripts\pythonw.exe main.py` 启动（此时从任务栏关闭）。

## 玩法

| 操作 | 效果 |
|------|------|
| 直接说话 | 语音识别 → 判断是否需要回应 → 气泡回复 |
| 说"看看我的屏幕" | 模型调用 `look_at_screen` 工具截屏并描述 |
| 单击桌宠 | 唤起/收起文字输入框，回车发送 |
| 双击桌宠 | "用户触碰了你"彩蛋 |
| 拖动 | 移动位置 |

## 项目结构

```
main.py        # 入口（Qt + asyncio 事件循环装配）
brain.py       # LLM 通信、消息打包、工具参数解析
vision.py      # 截屏（Windows ImageGrab；Wayland PipeWire → gnome-screenshot → X11 ImageGrab）
listen.py      # VAD 语音检测 + Whisper 转写
ui.py          # DeskFriend 窗口：气泡、输入框、动画、消息调度
pw_capture.py  # Wayland 静默截屏后端（ScreenCast Portal + PipeWire + GStreamer）
win_compat.py  # Windows 启动兼容（预加载系统 MSVC 运行库）
assets/        # 贴图 + VAD 模型
tests/         # pytest 用例
docs/DEVELOPMENT.md # 开发文档：技术决策、路线图、验收标准
```

## 开发

```bash
uv run ruff check .     # lint
uv run ruff format .    # 格式化
uv run pytest           # 测试
```

## 版本记录

当前版本 `0.1.0`，变更见 [CHANGELOG.md](CHANGELOG.md)。

## 安全

发现安全问题请不要开公开 Issue，按 [SECURITY.md](SECURITY.md) 的流程私下报告。

## License

本项目采用 [GPL-3.0](LICENSE) 许可证。由于依赖 PyQt5（GPLv3 / 商业双许可），分发时请遵守 GPLv3 条款。

Copyright (C) 2026 UESTC_DeskFriend contributors
