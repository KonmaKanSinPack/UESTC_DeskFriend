# UESTC_DeskFriend 开发文档

> 桌面 AI 伙伴（桌宠）项目。形象为糯糯（Q版弗洛洛），具备听觉（语音识别）、视觉（截屏理解）、对话（可插拔回复后端）能力。
> 本文档定义当前阶段的总体开发任务，作为后续迭代的基准。

## 1. 现状盘点

### 已有能力
- PyQt5 无边框透明置顶窗，可拖动，双击可交互
- Silero VAD 语音活动检测 + faster-whisper 中文转写
- `ImageGrab` 截屏 + base64 注入多模态 LLM 上下文
- OpenAI 兼容接口（`gemini-2.5-pro`），tool calling 循环（`look_at_screen`）
- 消息队列（生产者-消费者）+ 是否回复的预判断（`should_reply`）
- 配置外置：`config.toml`（`API_KEY` / `BASE_URL`）

### 已知问题
| # | 问题 | 严重度 |
|---|------|--------|
| P1 | `ImageGrab` 在 Wayland 下不可用，视觉模块整体失效 | 高 |
| P2 | Wayland 下 `self.move()` 无效，拖动失效 | 高 |
| P3 | 回复只 `print` 到终端，无 UI 气泡，交互不可见 | 高 |
| P4 | torch 依赖过重（仅用于 VAD），拖慢启动、占内存 | 中 |
| P5 | 录音逻辑缺陷：静音阈值仅约 640ms（20×32ms），说话稍停顿就会被截断；且录音无最长时限，长语音会一直缓冲 | 中 |
| P6 | 忙碌时新消息直接丢弃，无排队上限与合并策略 | 低 |
| P7 | 单文件 374 行，无模块拆分、无测试 | 中 |
| P8 | 无 `requirements.txt` / 环境说明，不可复现 | 中 |
| P9 | `pytesseract` 导入了但未使用 | 低 |
| P10 | 无记忆：上下文仅 `deque(maxlen=5)`，按条数截断且进程重启即失忆 | 高 |

## 2. 总体目标

在 **Ubuntu (GNOME / Wayland)** 上让桌宠完整可用：能听、能看、能以气泡形式回复，代码结构可持续演进。暂不做 Rust 迁移（理由：本项目是 LLM 编排器，瓶颈在推理后端与网络，Python 胶水层非性能瓶颈；分发需求出现前不重写）。

## 3. 技术决策（已确定）

1. **保持 Python**，不做语言迁移。
2. **截屏在 Wayland 下走 PipeWire**（ScreenCast Portal persist_mode=2 + GStreamer 按需拉帧，`pw_capture.py`）：首次运行弹一次 GNOME 授权框，之后完全静默；回退链为 PipeWire → `gnome-screenshot`（有闪光）→ `ImageGrab`（X11）。已验证的死路：GNOME Shell 私有 Screenshot D-Bus 对第三方 `AccessDenied`；XDG Portal Screenshot 接口非交互模式也弹框。注意 venv 需 `--system-site-packages` 以使用系统 gi/GStreamer。
3. **VAD 去 torch 化**：Silero VAD 改用 ONNX Runtime 推理，移除 torch 依赖。
4. **UI 增加对话气泡**：LLM 回复显示在桌宠旁的气泡中，超时自动消失。
5. **模块拆分**：按器官划分（`brain` / `vision` / `listen` / `ui`），`main.py` 只做装配。
6. **依赖管理用 uv**（`pyproject.toml` + `uv.lock`，Python 固定 3.12），**代码规范用 ruff**（lint + format，`line-length = 120`）。
7. **音频采集用 sounddevice**，替代 pyaudio，消除编译期依赖（portaudio 头文件）；Linux 下仍需系统运行库 `libportaudio2`（wheel 不捆绑 Linux 二进制）。
8. **回复后端可插拔，双实现由配置选择**（2026-08-10 更新，替代"不引入 AstrBot"旧结论）：
   - `brain.py` 只保留统一契约（`ReplyBackend` 抽象 + `BackendResponse` 响应对象），按 `config.toml` 的 `BACKEND` 键经工厂装配：
     - `astrbot`（默认）：走 OneBot 11 伪装通道接入 AstrBot 的桃桃，对话/记忆/人格由 AstrBot 接管，桌宠侧记忆为空操作
     - `openai`：直连 OpenAI 兼容接口 + 自建记忆系统（原方案，保留作对照与回退）
   - 自建记忆模块 `memory.py` 仍存在，服务 openai 后端：stdlib `sqlite3` 单文件（`pet.db`），不引入 ORM、不引入向量数据库。
   - 曾评估过的"自定义 AstrBot 平台适配器插件"（AstrBot ≥4.16 支持）列为阶段四候选，届时可消灭 OneBot 协议的静默窗口 hack 并支持流式回复。
9. **遗忘策略 = 轮次截断 + LLM 增量摘要**：超出轮次上限的老历史不硬丢，由 LLM 滚动生成摘要注入上下文（上下文 = system + facts + 摘要 + 最近 N 轮）。
10. **长期记忆用事实表 + LLM 抽取**：`facts` 表存结构化事实（如"用户在 UESTC 读书"），抽取/去重/更新/删除全部交给 LLM 输出 JSON 操作完成，不写自研相似度去重；facts 量小，全量注入 system prompt，不做检索。
11. **语义检索走降级路线**：优先 SQLite FTS5 关键词检索（内置、零依赖）；embedding top-k 仅列为远期候选（端点支持则用 API，否则本地小模型），不在初期目标内。

## 4. 分阶段任务

### Phase 0 — 工程基础（先做）
- [x] ~~补 `requirements.txt`~~ → 改用 uv（`pyproject.toml` / `uv.lock` / `.python-version`）
- [x] 引入 ruff 并跑通 `ruff check` + `ruff format`
- [x] 移除未使用的死导入（`pytesseract` / `requests` / `httpcore` / `xmlrpc` 等）
- [x] pyaudio → sounddevice，消除系统依赖
- [x] 配置改用 TOML（内置 `tomllib`，无第三方依赖），提供 `config.example.toml`
- [x] README 写清运行步骤

### Phase 1 — Wayland 兼容（核心）
- [x] 截屏：Wayland 走 `gnome-screenshot` 子进程 + X11 回退 `ImageGrab`（已在本机实测，2880×1800 全屏内容正确）
- [x] 验证 GNOME Wayland 下 `look_at_screen` 工具链路可用（已实测）
- [x] 拖动：XWayland 下 Qt 拖动正常（已实测）
- [x] 录音逻辑修复：静音阈值 640ms → 约 1.5s；新增最长录音约 15s 上限；空转写不再触发回复

### Phase 2 — 交互可见性
- [x] 气泡 UI：显示 LLM 回复文本，10s 自动隐藏；语音受理后显示"听到了，正在想…"，跳过/出错时收起
- [x] 消息队列策略：忙碌时入队（上限 20 条），消费者一次性合并积压的连续语音片段

### Phase 3 — 依赖瘦身与结构重构
- [x] VAD 迁移到 onnxruntime（提前完成）：`assets/silero_vad.onnx` 本地模型 + `SileroVadOnnx` 封装，删除 torch / `torch.hub.load`，启动零下载
- [x] 模块拆分：`main.py` 只做装配入口；`brain.py`（LLM + pack_msg + parse_tool_args）/ `vision.py`（截屏）/ `listen.py`（VAD + Whisper）/ `ui.py`（DeskFriend 窗口）
- [x] 最小测试：`tests/` 9 个用例（pack_msg、parse_tool_args 兜底、VAD 打分与状态重置），`uv run pytest` 全绿

### Phase 4 — 打磨（视情况）
- [x] 贴图状态动画（单图程序动画，整窗微动）：待机缓慢呼吸、思考快速晃动、说话弹跳；拖动时暂停
- [x] 文字输入：单击桌宠唤起输入框（回车发送进消息队列，Esc 收起），双击保留"触碰"彩蛋
- [ ] 打包分发（PyInstaller / AppImage），届时重新评估 Rust 收益

### Phase 5 — 记忆系统：持久化短期记忆 + 摘要压缩
目标：解决 P10，糯糯"记得今天聊过什么"，重启不失忆。
- [x] 新增 `memory.py`：`MemoryStore`（stdlib `sqlite3`，`pet.db`），`messages` 表（role / msg_json / turn_id / created_at），启动时加载未压缩历史
- [x] `brain.py`：`deque(maxlen=5)` 替换为 `MemoryStore` 驱动的上下文；消息按轮次（turn_id）归组（用户文本消息开启新一轮，tool/图片消息归入当前轮），不再按条数截断
- [x] 摘要压缩：未压缩轮次超过 `MAX_CONTEXT_TURNS`(30) 时，最老轮次经 LLM 增量更新滚动摘要（`summaries` 表单条记录），保留最近 `KEEP_RECENT_TURNS`(10) 轮；上下文组装为 system + 摘要 + 最近 N 轮；`ui.py` 在回复展示后触发，压缩失败不影响对话
- [x] 无需回复的听觉消息也落记忆：`brain.memorize()` 以"[背景谈话，无需回应]"标注独立成轮，参与上下文与压缩，但不生成回复
- [x] 测试：`tests/test_memory.py` 14 个用例（历史落库、重启恢复、摘要触发与增量更新、base64 图片落库脱敏）
- 验收：聊几轮后重启进程，糯糯仍能接续之前的话题

### Phase 6 — 记忆系统：事实型长期记忆
目标：糯糯"记得你这个人"——名字、学校、偏好、近期在意的事。
- [x] `facts` 表（id / content / created_at / updated_at）+ `meta` 表（记录抽取游标 `last_extracted_turn`）
- [x] 事实抽取：回复完成后触发（背景谈话攒批，随下次回复一并抽取，避免逐条背景消息浪费 LLM 调用）；LLM 输入"现有 facts + 最近 ≤20 轮"，输出 JSON 操作（add / update / delete）写回 `facts`；拿到响应即推进游标，坏输出丢弃本批不重试
- [x] system prompt 注入：facts 全量拼接在人设之后、摘要之前；条数超 `MAX_FACTS`(50) 时由 LLM 合并压缩至 30 条，坏输出不落库
- [x] 测试：facts CRUD、游标推进、抽取 JSON 容错（坏输出不落库）、三类操作与非法 id 校验、背景轮次覆盖、合并压缩、注入格式
- 验收：告诉糯糯一个个人信息（如"我下周三要交实验报告"），隔天重启后主动问起或能答出

### Phase 7 — 记忆系统：历史语义检索（可选，视 Phase 5/6 手感）
- [ ] SQLite FTS5 对 `messages` 建索引，按当前输入检索相关历史轮次注入上下文
- [ ] （远期候选）embedding top-k 检索：端点支持 embedding 则用 API，否则本地 bge-small-zh
- 验收：翻旧账类问题（"我上次说的那本书叫什么"）能答对

## 5. 验收标准（总任务完成定义）

1. 在 GNOME Wayland 会话中运行，语音说一句话 → 桌宠气泡给出回复，全程不依赖终端输出。
2. 说"看看我的屏幕"类指令 → 模型成功调用 `look_at_screen` 并描述屏幕内容。
3. 全新环境按 README 可复现运行。
4. 依赖中无 torch；冷启动时间较现状明显下降。

## 6. 非目标（本阶段不做）

- Rust 重写
- 多平台（Windows/macOS）适配
- 向量数据库 / embedding 检索（仅 Phase 7 远期候选）
- TTS 语音合成（可列为远期候选）

> 注：原"引入外部 Agent 框架（如 AstrBot）"一项已随技术决策 8 的更新而取消——AstrBot 现在是默认回复后端。

## 7. AstrBot 接入（C方案，2026-08-10）

### 7.1 目标与路线

把桌宠的"脑子"从"自己调 LLM"换成可插拔后端：`astrbot`（默认，走 OneBot 11 伪装通道接入 AstrBot 的桃桃，记忆/人格由 AstrBot 全局记忆接管）与 `openai`（原直连 LLM + 自建记忆，保留作对照与回退）。

**技术路线**：桌宠伪装成一个"OneBot 实现"（迷你 NapCat），反向 WebSocket 连上 AstrBot 的 OneBot 适配器；消息流：桌宠说话 → 伪装 OneBot 反向推送（`message` 事件）→ AstrBot 对话流（记忆+人格）→ 桃桃回复（`send_private_msg` 动作）→ 回传桌宠显示。身份：sender 固定映射 `1063310598`（桃桃认得老公，解锁完整人格）。

### 7.2 模块结构（深度解耦）

```
ui.py ──5接口──▶ brain.py(Brain门面) ──▶ backends/{base,openai,astrbot}.py
                                            │  BACKEND 配置选择
   astrbot: AstrBotBackend ──▶ onebot_bridge.py ──WS+token──▶ AstrBot
   openai:  OpenAIBackend   ──▶ openai SDK + memory.py(pet.db)
```

- `backends/base.py`：`ReplyBackend` 抽象基类（`get_llm_response` / `get_response_with_context` / `memorize` / `maybe_compress` / `maybe_extract_facts`）+ 统一响应对象 `BackendResponse`（`content` + `tool_calls`），ui 只认契约，不认 SDK 类型
- `backends/__init__.py`：`create_backend(config)` 工厂，按 `BACKEND` 键装配
- `backends/openai.py`：旧直连 LLM 逻辑整体迁移（tools 循环、摘要压缩、事实抽取）
- `backends/astrbot.py`：OneBot 通道对话 + 屏幕感知状态机 + `[look_at_screen]` 文本指令协议；should_reply 判定由 `LLMJudge`（judger.py）统一负责；记忆三接口为空操作
- `onebot_bridge.py`：OneBot 11 反向 WS 客户端（Bearer token、30s 心跳、动作响应、静默窗口结算）

### 7.3 关键机制

- **静默窗口**：AstrBot 可能连发多条回复（如"回复中提示"占位 + 最终回复），收到回复后 `ASTRBOT_SETTLE` 秒内无新回复才视为最终回复，取最后一条
- **屏幕感知状态机**（事件驱动 + 变化门控 + 冷却）：idle 60s / active 20s 截屏 → 16×16 感知哈希 diff → 变化幅度超阈值 且 冷却期过 且 不在对话中 且 用户静默期过，才发桃桃"主动观察"消息；回复"无"类则静默丢弃，有内容则主动冒泡（`reply_sink` 回调）
- **`[look_at_screen]` 协议**：桃桃回复含此标记 → 桌宠本地截屏 → 附图追问（≤3 轮）→ 显示最终回复；作为"LLM 主动调工具看屏幕"的轻量实现（MCP 为阶段四候选）
- **should_reply 判定（2026-08-10 起统一走 LLM 决策器）**：废弃本地关键词规则（漏判严重，且提示词与实现分离）。`backends/judger.py` 的 `LLMJudge`（few-shot 提示词 + "不确定输出 true"）统一判定：
  - openai 后端走通用 LLM 通道（context 里的共享提示词）
  - astrbot 后端由桌宠侧自调 LLM（复用 `API_KEY`/`BASE_URL` + `JUDGE_MODEL` 配置），判定**不进 AstrBot 对话流**，不污染桃桃记忆
  - 判定器未配置 / 调用失败 → 回退默认回复（宁可多回，不可漏听）

### 7.4 AstrBot 侧依赖（2026-08-10 已实测联调成功）

AstrBot 的 OneBot V11 适配器即 **aiocqhttp 库**（Quart ASGI，服务器 banner 为 hypercorn），
反向 WS 端点 `/ws`、`/ws/event`、`/ws/api`。握手**强制要求三个头**（`_handle_wsr` 源码）：

1. `Authorization: Bearer <token>` —— 正则 `(?:[Tt]oken|[Bb]earer) (\S+)` 校验，值必须等于适配器配置的 token
2. `X-Client-Role: universal` —— 连接角色（event/api/universal），**缺了直接 400**；
   桌宠用 universal（同时上报事件 + 接收动作）
3. `X-Self-ID: <self_id>` —— 机器人身份，服务器按它路由动作下发（允许 `*` 通配）

联调踩坑记录：只发 Bearer 头 → 400（不是 token 错了，是缺 X-Client-Role 触发 KeyError）；
无头 → 401；`?access_token=` 查询参数一律 401（该库只认 Authorization 头）。
AstrBot 侧配置：反向 WebSocket 主机 `0.0.0.0`、端口自定、token 自定；NapCat 等正常客户端
与桌宠可共存（单连接时动作路由直接用唯一连接）。

「回复中提示」：实测桃桃单条回复即结束（未见占位提示），`ASTRBOT_SETTLE = 2.0` 默认值即可；
若日后开启提示且间隔超过 settle，需相应调大（见 onebot_bridge 静默窗口语义）。
