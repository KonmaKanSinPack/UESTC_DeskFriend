# technical — 技术规格权威（骨架版）

> **接口/数据结构变更时先改这里、再改码**（RULES §5、§8）。
> 当前为最小骨架（2026-08-21 建立）：从 DEVELOPMENT §7/§8、ARCHITECTURE.md 与代码签名
> 抽查提炼，保证与现状一致；细节随接口变更增量充实，不追求一次写全。
> 架构分层与硬规则见 `ARCHITECTURE.md`（本文不重复）。

## 1. 模块地图

```
main.py        装配入口（onnxruntime 预载 → QApplication+qasync → Skin+Spine）
spine.py       主控中枢 Spine（plain class）：接线七条（listen.mouth 注入、interrupt_requested、
               text_signal、text_submitted、touched、quit_requested、brain.observe_sink）
               + 消息队列（(source, text) 分流）+ should_reply + do_response
brain.py       Brain 门面：统一契约转发后端 + pack_msg/parse_tool_args 纯函数
backends/      base.py（契约）/ __init__.py（工厂+create_judge）/ astrbot.py / openai.py / judger.py
onebot_bridge.py  OneBot 11 反向 WS 伪装客户端
listen.py / mouth.py / vision.py / skin.py   器官（耳/嘴/眼/皮）
psd_renderer.py  分层渲染器（皮的内部细节，2026-09-04）：SPRITE 指向 manifest.json 时
               启用——眨眼/视线/口型/呼吸/发摆；渲染器自驱动画，窗口恒定不动
tools/         prepare_psd（See-through PSD→拆 L/R+扩边+manifest）/ make_demo_pet（演示层）
asr.py         ASR 引擎层（对称 tts.py）：SenseVoice（sherpa-onnx 本地，默认）/ Whisper（回退）
               + create_asr 工厂 + 模型自动下载（hf-mirror 直链 → GitHub tar 兜底）
tts.py         TTS 抽象 + SiliconFlow / CosyVoice2(未完成) / Dummy 工厂
memory.py      MemoryStore（sqlite3 单文件 pet.db，服务 openai 后端）
pw_capture.py  Wayland 静默截屏后端（ScreenCast Portal + PipeWire + GStreamer）
```

## 2. spine ↔ brain 契约（ReplyBackend，backends/base.py）

```python
class ReplyBackend(ABC):
    async def get_llm_response(self, message, model=None) -> BackendResponse   # 单条入（可带 tool 结果/附图）
    async def get_response_with_context(self, context, model=None, use_tools=False) -> BackendResponse
    def memorize(self, message)                    # 背景谈话只落库不回复（openai 实现；astrbot 空操作）
    async def maybe_compress(self)                 # 滚动摘要触发（openai 实现；astrbot 空操作）
    async def maybe_extract_facts(self)            # 事实抽取触发（同上）

@dataclass
class BackendResponse:
    content: str
    tool_calls: list[ToolCall] = []
    answered: bool = True   # False = 桥超时兜底文案 → spine 只显示不朗读（防回声环）
    speak: bool = True      # False = 只显示不朗读（主动观察空闲期：仅活跃期朗读，2026-09-03）
```

- 工厂 `create_backend(config)`：按 `BACKEND` 键装配（`astrbot` 默认 / `openai`）。
- 判定器 `create_judge(config)` → `LLMJudge`（judger.py，few-shot 提示词 + 不确定输出 true）；
  `JUDGE_URL` 优先、缺则回退 `BASE_URL`。
- **判定路由**（openai 后端，2026-08-14）：`get_response_with_context` 上下文首条 system ==
  `JUDGE_SYSTEM_PROMPT`（spine 构造，单一来源）→ 走 judge 通道，不进主对话流；
  记忆链路（摘要/抽取/合并）各有自己的提示词，不被误路由。
- **主动观察统一消息流**（2026-09-03）：astrbot 后端观察门控通过后经 `observe_sink`
  回调（Brain 门面 property → 后端属性）把观察文案交 spine 入队；消息队列元素为
  `(source, text)`（`user`/`proactive`），consumer 只合并连续 user 源、proactive 独立
  成条且跳过 should_reply（门控已在后端）。呈现统一走 `do_response`：空 content =
  完全静默（后端对"无"类回复返回空串，仅主动观察会出现）；`speak=False` 只冒泡不朗读。
  自发消息不更新用户时间窗、不消费打断标记（`PROACTIVE_MARKER` 前缀识别分支）。

## 3. 器官接口（对 spine 的边界）

| 器官 | 命令（被主控调） | 状态/信号（对外） |
|---|---|---|
| 耳 listen | `listen.mouth = mouth`（单向只读注入） | `text_signal(str)`、`interrupt_requested` |
| 嘴 mouth | `async speak(text)`、`async interrupt() -> 前缀`、`async stop()`（退出收尾） | `busy` / `speaking` / `playing` / `window_open`、`finished` |
| 眼 vision | `look_at_screen()`（内部多后端回退） | — |
| 皮 skin | `show_bubble(text, timeout_ms)` / `hide_bubble()` / `set_anim_state(state)` / `set_pending(bool)`（生成中三点指示，2026-09-03） | `text_submitted(str)`、`touched`、`quit_requested` |

- **ASR 引擎**（2026-09-03，asr.py）：`ASR.transcribe(audio_f32, sample_rate) -> str`
  （同步，跑耳线程）。`create_asr(config)` 按 `ASR_BACKEND` 装配：`sensevoice`
  （sherpa-onnx 本地，默认，模型缺失自动下载）→ 失败回退 `whisper`（faster-whisper
  small + initial_prompt）。器官内部换件，`text_signal` 接口不变。
- **采集侧三小修**（listen.py 纯函数）：`PreRollBuffer`（触发前 ~224ms 保首字，
  回声块不入缓冲）、`trim_trailing_silence`（裁尾部静音治幻觉，保留 3 块停顿）、
  `boost_if_quiet`（峰值 <0.25 才等比放大）。已知限制：SenseVoice 转写无标点
  （仅喂 LLM 不朗读，链路无影响）。

- **远程工具链**（2026-09-09，AstrBot 插件配套）：`spine.execute_tool(name, args)
  -> (text, image_data-url|None)` 为工具执行**唯一分发点**（openai 的 tool_executer
  是它的回喂打包 wrapper）。AstrBot 侧插件经 OneBot action `deskfriend_tool`
  `{tool, args}` → 桥 `tool_handler` → astrbot 后端 `tool_sink`（brain 门面）→
  execute_tool；回包 `{status, text, image}` 按 echo 返回。新工具只在 execute_tool
  加分支，两个后端 + 插件同时获得。
- **退出编排**（2026-09-03）：皮 `quit_requested`（托盘/右键菜单「退出」广播）→
  `spine._shutdown()`（幂等闩）：`brain.stop()`（结算桥在飞请求）→ `mouth.stop()`
  （停朗读 + TTS close）→ `quit_app()`（main 注入 `app.quit`，spine 零 Qt）。
  窗口带 `Qt.Tool`（不进任务栏/Alt-Tab）。

- TTS 抽象（tts.py）：`async speak(text)` / `async interrupt() -> str` / `busy` / `playing` /
  `close()`；`sentence_done_callback` 钩子驱动句间监听窗口（Mouth._sentence_gap）。
- 打断链路：耳 emit `interrupt_requested` → spine → `mouth.interrupt()` 得前缀 →
  `brain.set_interruption(prefix)` 注入「[对话被打断] …」下次对话。

## 4. 数据结构

- **memory.py 四表**（sqlite3，`pet.db`）：`messages`（turn_id 归组、summarized 标记）、
  `summaries`（恒单行滚动摘要）、`facts`（长期事实 CRUD）、`meta`（抽取游标）。
  核心不变式：**未压缩消息与内存 context 一一对应**（条数/顺序）。落库过
  `sanitize_message`（base64 图片→占位符）；**恢复过 `revive_message`**（2026-09-04：
  非 `data:` 的图片段→`[历史截图，内容已省略]` 文本段——占位符留在 image_url 里
  发 API 会被 base64 解码 500，毒化重启后所有对话）。
- **config 分层（2026-09-04）**：`config_loader.load_config()` 合并 `config/common.toml`
  （BACKEND 选择器 + 共享：JUDGE_*/TTS_*/ASR_*/SPRITE/MEMORY_*）与
  `config/{BACKEND}.toml`（openai：API_KEY/BASE_URL/OPENAI_MODEL/SYSTEM_PROMPT；
  astrbot：ASTRBOT_*/SCREEN_*），后端键覆盖同名共享键；缺文件抛 `ConfigError`
  带修复指引。记忆阈值默认值在 `backends/openai.py` 的 `DEFAULT_*`（30/10/50），
  经 MEMORY_MAX_TURNS / MEMORY_KEEP_RECENT / MEMORY_MAX_FACTS 覆盖。
  权威键清单见 `config/*.example.toml`。

## 5. OneBot 桥协议（onebot_bridge.py ↔ AstrBot）

- 反向 WS 连 AstrBot OneBot 适配器（`ASTRBOT_WS_URL`，aiocqhttp/Quart）。
- **握手强制三头**：`Authorization: Bearer <token>`、`X-Client-Role: universal`、
  `X-Self-ID: <id>`（缺 X-Client-Role 直接 400；`?access_token=` 参数一律 401）。
- 动作流：桌宠发 `message` 事件（sender 固定 `ASTRBOT_USER_ID`）→ 收 `send_private_msg` 动作 = 回复。
- **静默窗口结算**：收到回复后 `ASTRBOT_SETTLE` 秒无新回复视为最终回复（取最后一条）。
- **屏幕感知**（astrbot 后端）：idle 60s / active 20s 周期截屏 → 16×16 感知哈希 diff →
  超阈值+冷却过+非对话中+用户静默 → 发「主动观察」消息；回复含 `[look_at_screen]` →
  本地截屏附图追问（≤3 轮）。

## 6. 路线图（占位）

Phase 4 打包分发（PyInstaller）、Phase 7 FTS5 历史检索、AEC 全双工轮——状态见 `PROGRESS.md` 待办。
