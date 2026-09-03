# UESTC_DeskFriend

桌面 AI 伙伴（桌宠），形象为糯糯（Q版弗洛洛）。常驻桌面一隅，能听、能看、能聊、能说：

- **听觉**：麦克风常驻监听，Silero VAD（ONNX，本地模型）检测说话，faster-whisper 中文语音转文字
- **视觉**：截屏注入多模态大模型上下文，说一句"看看我的屏幕"它就能描述你在干什么
- **对话**：可插拔回复后端（`config.toml` 的 `BACKEND` 键选择）：
  - `astrbot`（默认）：经 OneBot 11 伪装通道接入 AstrBot 的桃桃，记忆/人格由 AstrBot 接管；自带屏幕感知（屏幕变化时主动观察冒泡）
  - `openai`：直连 OpenAI 兼容接口（默认 `gemini-2.5-pro`），自建 SQLite 记忆
- **语音输出**：桃桃回复自动朗读（SiliconFlow 托管 CosyVoice2，可克隆自定义音色）；朗读中你说话会立即打断，桃桃知道说到哪
- **交互**：语音 / 单击输入文字 / 双击"触碰"彩蛋 / 拖动摆放，带呼吸、思考、说话的状态动画

## 运行环境

- **Windows 11**（已验证）与 **Linux**（开发环境为 Ubuntu GNOME **Wayland**；X11 亦可，功能降级见下）
- Python 3.12（由 uv 管理）
- 可选 NVIDIA GPU（Whisper 自动检测，无显卡回退 CPU int8）

### 系统依赖（仅 Linux）

```bash
sudo apt install -y libportaudio2 libxcb-xinerama0
# 静默截屏（Wayland）需要：
sudo apt install -y python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-pipewire
```

- `libportaudio2`：音频采集（sounddevice）
- `libxcb-xinerama0`：PyQt5 xcb 插件运行库
- gi/GStreamer 三件套：Wayland 下 PipeWire 静默截屏。缺失时回退 `gnome-screenshot`（有闪光灯效）
- Windows 无需任何系统依赖；截屏自动走 Pillow `ImageGrab`，PipeWire 后端不会加载

### 安装

**Windows：**

```bash
uv sync
```

**Linux：**

```bash
# venv 必须带 --system-site-packages，否则用不上系统的 gi（静默截屏失效）
uv venv --system-site-packages --python 3.12
uv sync
```

### 配置

```bash
cp config.example.toml config.toml
```

**跑通所需键位清单**（按功能勾选，缺哪个功能就填哪组）：

| 功能 | 必填键 | 说明 |
|---|---|---|
| 对话（astrbot 后端，默认） | `ASTRBOT_WS_URL` / `ASTRBOT_WS_TOKEN` | AstrBot 反向 WS 地址与 token，见下节 |
| 判定器（该不该回） | `API_KEY` / `BASE_URL` + `JUDGE_MODEL` | 判定走 LLM（本地 Ollama / 云端均可） |
| 语音输出（朗读） | `TTS_BACKEND` / `TTS_API_KEY` / `TTS_VOICE_REF` / `TTS_VOICE_REF_TEXT` | 见"语音输出"章节 |
| 可选 | `SYSTEM_PROMPT` 人设 / `SPRITE` 贴图 / `SCREEN_*` 屏幕感知参数 | 不填用默认 |

> 注意：判定器与 openai 后端共用 `API_KEY`/`BASE_URL`；若判定器用本地模型（如
> Ollama/LM Studio），直接把 `BASE_URL` 指向本地端点即可。

### 接入 AstrBot（astrbot 后端）

桌宠伪装成一个 OneBot 实现（迷你 NapCat），反向 WebSocket 连上 AstrBot 的 OneBot V11
适配器（aiocqhttp），消息流：桌宠说话 → 事件上报 → AstrBot 对话流（记忆+人格）→
桃桃回复 → 回传气泡显示。sender 固定为 `ASTRBOT_USER_ID`（默认 1063310598，桃桃认得的"老公"）。

**AstrBot 侧**（WebUI → 消息平台 → 添加 aiocqhttp 适配器）：启用反向 WebSocket，
主机 `0.0.0.0`、端口任意（如 8786）、token 自定；NapCat 等正常客户端照常连接，
桌宠与之共存。

**桌宠侧**：config.toml 填好 `ASTRBOT_WS_URL`（如 `ws://192.168.10.2:8786/ws`）与
`ASTRBOT_WS_TOKEN` 即可。握手自动携带三个头（aiocqhttp 强制要求，缺一不可）：
`Authorization: Bearer <token>`、`X-Client-Role: universal`、`X-Self-ID: <self_id>`。

**屏幕感知**：idle 态每 60s / 对话活跃期每 20s 截屏做感知哈希对比，屏幕显著变化且
通过冷却/静默门控时，发桃桃"主动观察"消息；桃桃觉得值得说会主动冒泡显示气泡。

### 语音输出（TTS）

桃桃的回复会自动朗读。方案选择（`TTS_BACKEND`）：

| 后端 | 说明 |
|---|---|
| `siliconflow`（默认） | SiliconFlow API 托管的 CosyVoice2，稳定 1~2s/句，支持音色克隆 |
| `cosyvoice2` | 本地推理（**未完成**：对依赖版本栈敏感，官方环境验证正常但需独立环境常驻，暂不推荐） |
| `dummy` / `none` | 不发声（占位 / 关闭） |

**最小配置**（siliconflow）：

```toml
TTS_BACKEND = "siliconflow"
TTS_API_KEY = "sk-..."        # SiliconFlow 密钥（https://cloud.siliconflow.cn）
TTS_VOICE_REF = "assets/vo_hutao_mimitomo_friendship2_01.wav"  # 参考音频（音色样本）
TTS_VOICE_REF_TEXT = "来得正好。我刚想找人闲聊你就出现了。该不是有什么心灵感应吧。"  # 音频文本
```

**换音色指南**（克隆任意音色）：

```
1. 准备 3-10 秒干净人声（无 BGM/混响），放到 assets/
2. 转写参考音频文本（可用 faster-whisper，见下），并人工逐字核对
   —— 文本必须与音频内容精确对应，错一个字克隆质量就崩（本项目踩过坑）
3. 把 TTS_VOICE_REF 指向音频路径、TTS_VOICE_REF_TEXT 填入核对后的文本，重启生效
```

转写参考音频（示例）：

```bash
uv run python -c "from faster_whisper import WhisperModel; m = WhisperModel('small', device='cuda', compute_type='float16'); print(''.join(s.text for s in m.transcribe('assets/参考音频.wav', language='zh')[0]))"
```

**打断机制**：朗读中你说话/打字 → 朗读立即停止 → 下一条消息自动带上
「[对话被打断] 你刚才说到『…』处被打断了」→ 桃桃知道说到哪，自然衔接。

### 模型

- VAD 模型已内置（`assets/silero_vad.onnx`），无需下载
- faster-whisper `small` 首次使用需下载（约 460MB）。之后启动全部离线
- 网络不佳时用镜像下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
uv run python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

### 运行

```bash
uv run python main.py
```

- Linux Wayland 首次运行会弹一次 GNOME 屏幕共享授权框（PipeWire 截屏），点允许后永久静默，授权 token 存于 `pw_restore_token`（删除该文件可重新授权）。`Ctrl+C` 退出。
- Windows 无此授权流程；`Ctrl+C` 退出。

## 玩法

| 操作 | 效果 |
|------|------|
| 直接说话 | 语音识别 → 判定器判断该不该回 → 气泡回复 + 朗读 |
| 说"看看我的屏幕" | 桌宠本地截屏，把屏幕图发给桃桃描述 |
| 桃桃回复时 | 自动朗读（可打断：说话即停，桃桃知道说到哪） |
| 单击桌宠 | 唤起/收起文字输入框，回车发送 |
| 双击桌宠 | "用户触碰了你"彩蛋 |
| 拖动 | 移动位置 |
| 屏幕有变化（空闲时） | 桃桃主动观察，有值得说的会主动冒泡 |

## 项目结构

```
main.py        # 入口（Qt + asyncio 事件循环装配）
brain.py       # Brain 门面：统一契约，转发给后端
backends/      # 回复后端：base（契约）/ astrbot（OneBot 通道+屏幕感知）/ openai（直连+自建记忆）
               # + judger.py（should_reply 判定器，LLM）
onebot_bridge.py  # OneBot 11 反向 WS 伪装客户端（事件上报/动作响应/静默结算）
tts.py         # 语音输出：TTS 抽象 + SiliconFlow（默认）/ CosyVoice2（未完成）/ Dummy
vision.py      # 截屏（Wayland PipeWire → gnome-screenshot → X11 ImageGrab）
listen.py      # VAD 语音检测 + SenseVoice/Whisper 转写（asr.py 引擎层：可插拔 + 模型自动下载）
asr.py         # ASR 引擎层：SenseVoice（本地 sherpa-onnx，默认）/ Whisper（回退）+ 工厂
skin.py        # 外观器官（皮）：贴图、气泡、输入框、动画、拖动/双击/右键输入、系统托盘与退出
spine.py       # 主控中枢（脊髓）：器官装配、消息队列、判定、回复与退出编排
pw_capture.py  # Wayland 静默截屏后端（ScreenCast Portal + PipeWire + GStreamer）
assets/        # 贴图 + VAD 模型 + 音色参考音频
tests/         # pytest 用例
docs_agent/    # agent 层文档：RULES（工作方式）、ARCHITECTURE（架构硬规则）、DEVELOPMENT、technical、PROGRESS、session/
docs_human/    # 人类可读层：design（设计愿景）、overall（全局技术文档）
```

## 开发

```bash
uv run ruff check .     # lint
uv run ruff format .    # 格式化
uv run pytest           # 测试
```
