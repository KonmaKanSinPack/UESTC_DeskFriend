# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 和
[语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-11-09

首个公开版本。

### Added

- 桌面 AI 伙伴「糯糯」：PyQt5 无边框透明置顶窗口，支持拖动、单击输入、双击互动和呼吸/思考/说话状态动画。
- 听觉：Silero VAD（ONNX 本地模型）+ faster-whisper 中文语音转写。
- 视觉：截屏注入多模态 LLM，支持 `look_at_screen` 工具调用。
- 对话：OpenAI 兼容接口，支持 tool calling 和气泡回复。
- 记忆：SQLite 持久化短期记忆、滚动摘要和事实型长期记忆。
- Windows 10/11 支持：
  - `ImageGrab(all_screens=True)` 多显示器截屏；
  - 约束 Windows 使用 PyQt5-Qt5 5.15.2（最后一个提供 Windows wheel 的版本）；
  - 预加载系统 MSVC 运行库，修复 PyQt5 与 onnxruntime 的 DLL 冲突；
  - Whisper CUDA 初始化失败时回退 CPU int8；
  - Windows 下启用 Qt5 高 DPI 缩放。
- 文档：Windows 安装/运行说明、安全策略、GPL-3.0 许可证。

### Changed

- 配置文件 `config.toml` 和 `pet.db` 按项目目录定位，支持从快捷方式或其他工作目录启动。
- `requires-python` 统一为 `>=3.12`。

### Security

- 新增 `SECURITY.md`，提供 GitHub 私密漏洞报告流程。
- 项目采用 GPL-3.0 许可证，与 PyQt5 的 GPLv3 许可兼容。
