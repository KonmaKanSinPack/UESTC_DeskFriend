# 魔改历程：UESTC_DeskFriend 开发总结

从一个 374 行的单文件原型，到一个能听、能看、有记忆、有人设的桌面 AI 伙伴。
本文按主题回顾每处关键改动：**问题是什么 → 实现思路 → 关键代码**。

## 1. 工程化地基

### 1.1 uv + ruff + 模块拆分

**问题**：无依赖清单、无代码规范、所有代码塞在 `main.py`。

**思路**：uv 管理依赖（`pyproject.toml` + `uv.lock` + `.python-version` 锁 3.12），ruff 做 lint+format；
按"器官"拆模块——`main.py` 只做装配入口，`brain`（LLM）/ `vision`（截屏）/ `listen`（听觉）/ `ui`（窗口）。
拆分本身零行为变化，顺手把工具参数解析抽成纯函数 `parse_tool_args` 便于测试。

### 1.2 pyaudio → sounddevice

**问题**：pyaudio 需要编译，uv 环境下缺 `portaudio.h` 直接装不上。

**思路**：sounddevice 纯 wheel、cffi 动态加载运行库，消除编译期依赖（Linux 仍需 `libportaudio2` 运行时）。
采集接口从 pyaudio 流换成 `sd.RawInputStream`，参数一一对应：

```python
# listen.py:46-52
self.stream = sd.RawInputStream(
    samplerate=self.SAMPLE_RATE,  # 16000 Hz
    channels=1,  # 单声道
    dtype="int16",  # 16位深度
    blocksize=self.CHUNK,  # 每次读取 512 帧
)
```

### 1.3 YAML → TOML

**思路**：Python 3.11+ 内置 `tomllib`，零第三方依赖，顺手删掉 PyYAML。配置只有 `API_KEY` / `BASE_URL`
两个键时，TOML 足够且更不"反人类"。后来 `SYSTEM_PROMPT` / `SPRITE` 也进了同一个 `config.toml`。

## 2. Wayland 生存战（最曲折的一章）

### 2.1 截屏四连试错

**问题**：Pillow 的 `ImageGrab` 在 Wayland 下根本不可用，桌宠的"眼睛"瞎了。

**试错链**（每条都真机验证过）：

1. `org.gnome.Shell.Screenshot`（GNOME Shell 私有 D-Bus）→ 新版 GNOME 对第三方返回 `AccessDenied`
2. XDG Portal `Screenshot` 接口（`interactive=false`）→ GNOME 照样弹授权框，且 dbus-next
   解析 portal 内省 XML 还有 bug（`power-saver-enabled` 属性名带连字符不合法）
3. `gnome-screenshot` 子进程 → 能用、无弹窗，但每次截屏**闪白光**，10 秒一次的定时截图把用户闪出戏
4. **终局：ScreenCast Portal（persist_mode=2）+ PipeWire + GStreamer**——授权一次永久静默

**思路**：走 OBS 同款方案。Portal 申请一条持久屏幕流，拿到 PipeWire node 和 fd，
GStreamer 建管线按需拉帧。restore_token 存本地 `pw_restore_token`，下次启动免授权。

```python
# pw_capture.py:152-156（SelectSources 关键参数）
options = {
    "handle_token": GLib.Variant("s", token),
    "types": GLib.Variant("u", 1),  # 只共享显示器
    "multiple": GLib.Variant("b", False),  # 只要一路流
    "persist_mode": GLib.Variant("u", 2),  # 记住授权，以后不再弹窗
}
```

**省电设计**：管线平时停在 PAUSED（实测空闲 CPU 0%），抓帧才 PLAYING，拉一帧立刻回去：

```python
# pw_capture.py:82-86
with self._grab_lock:
    self._pipeline.set_state(Gst.State.PLAYING)
    self._appsink.try_pull_sample(0)  # 丢掉暂停前可能残留的过期帧
    sample = self._appsink.try_pull_sample(2 * Gst.SECOND)
    self._pipeline.set_state(Gst.State.PAUSED)
```

### 2.2 闪光三连修

切到 PipeWire 后用户仍看到闪光，连续修了三次才彻底：

**第一坑（回退太积极）**：PipeWire 初始化有窗口期，初版窗口期内回退 gnome-screenshot → 闪。
修法：窗口期**宁可跳过不截**，也不许闪光。

**第二坑（pending 顺序 bug）**：跳过后每个进程的第一张截图仍闪——`pending` 只覆盖"已开始初始化"，
而初始化恰好在首次 `grab()` 里才启动，顺序错了。修法：`pending` 改为覆盖"未启动"状态，判断前先 `start()`。

**第三坑（有限等待）**：用户反问"为什么不直接等初始化完成？"。答案是可以等，但必须有界——
截屏在 Qt 主线程被 10 秒定时器调用，而授权框可能等用户几分钟，无限等会冻结 UI。
终版：`grab(init_wait=3.0)`，正常冷启动（token 已存在）0.1 秒出图，等授权则超时跳过。

```python
# vision.py:57-66（最终回退链）
if is_wayland:
    backend = _get_pw_capture()
    if backend is not None:
        backend.start()  # 触发惰性初始化（幂等）
        image = backend.grab()  # 有限等待初始化完成（默认 3s）
        if image is not None:
            return image
        if backend.pending or backend.ok:
            return None  # 初始化超期或抓帧失败：跳过本次，不闪白光
        # 初始化失败：才回退 gnome-screenshot
```

### 2.3 拖动与 Ctrl+C

- Qt 在 Wayland 会话下自动走 XWayland（xcb），`self.move()` 可用，拖动天然保留——这也是
  不强制 `QT_QPA_PLATFORM=wayland` 的原因
- Ctrl+C 无法退出：Qt 事件循环不返回 Python 解释器，SIGINT 处理器永远排队。
  修法：`signal.signal(signal.SIGINT, signal.SIG_DFL)`，让内核直接杀进程

## 3. 听觉链路

### 3.1 VAD 去 torch 化与 64 点上下文坑

**问题**：torch 仅用于 Silero VAD 却占几百 MB 内存，且 `torch.hub.load` 首次运行要从 GitHub 拉模型（国内很慢）。

**思路**：silero-vad 官方包自带 ONNX 模型，抽出 `silero_vad.onnx`（2.3MB）放进 `assets/`，
用 onnxruntime 直接推理，torch 整个移除。

**踩的坑**：迁移后 VAD 对任何输入（包括真实语音）输出恒为 0.001——Whisper 交叉验证确认音频链路正常，
才定位到 **v5 模型要求每块输入前面拼接上一块末尾 64 个采样点作为上下文**（v4 无此要求，官方包装器里有，
自己写就漏了）：

```python
# listen.py:32-37
def __call__(self, chunk_f32):
    """chunk_f32: (512,) 的 float32 音频块，返回语音概率。"""
    x = np.concatenate([self.context, chunk_f32[None, :]], axis=1)
    out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
    self.context = x[:, -self.context_size :]
    return float(out[0, 0])
```

修复后实测：静音 0.01，说话 0.7~1.0。

### 3.2 Whisper 两个运行时坑

**CUDA 硬编码**：原代码写死 `device="cuda"`，但开发机是纯 Intel 核显，直接起不来。
修法：`ctranslate2.get_cuda_device_count()` 自动检测，无卡回退 `cpu/int8`（实测 3 秒音频 0.9 秒转写，够用）。

**启动卡死**：即使模型已缓存，faster-whisper 每次启动都先连 HuggingFace 校验元数据，
网络不通时死在 TCP 连接上，终端停在"Whisper 推理设备"不动。修法：离线优先，本地无缓存才在线下载：

```python
# listen.py:69-76
try:
    # 优先离线加载本地缓存：在线模式即使模型已缓存也会先连 HF 校验，
    # 网络不通时会卡死在 TCP 连接上
    self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type, local_files_only=True)
except Exception:
    print("本地未找到 Whisper 模型缓存，转为在线下载...")
    self.whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
```

### 3.3 录音分段策略

VAD 打分 ≥0.5 触发录音；连续静音 45 块（约 1.5s，原来是 640ms，说话稍停顿就被截断）视为说完；
470 块（约 15s）硬性上限防缓冲无限增长；空转写不上报（避免拿空消息烦 LLM）。

## 4. 对话质量

### 4.1 should_reply 误判

**问题**：用户说"看看我的屏幕"，判定结果是"不需要回复"。

**根因**：判定调用也带了 `tools`——模型可能不回 `true`/`false` 而是直接发起工具调用，
content 为空，兜底成 False。修法：判定类调用一律不带 tools，并改进提示词（直接指令一律 true，背景碎片才 false）：

```python
# brain.py:277-282
kwargs = {}
if use_tools:
    # 判定类调用（如 should_reply）不能带 tools：
    # 模型可能直接发起工具调用而不输出文本，导致判定落空
    kwargs = {"tools": self.tools, "tool_choice": "auto"}
```

### 4.2 tool calling 400 三连

**问题**：走工具调用就报 `400 Invalid request`。实际是三个叠加的消息结构问题：

1. **套娃**：`get_llm_response(tool_msg)` 把已打包的 tool 消息又包了一层 `{"role": "user", "content": <dict>}`
2. **丢字段**：assistant 响应存 context 时丢了 `tool_calls`，后续 tool 消息找不到对应调用
3. **顺序错**：截图消息插在 assistant(tool_calls) 和 tool 之间，违反"tool 必须紧跟 tool_calls"

修法：`get_llm_response` 接受 str / dict / list 三种入参，存 context 保留 tool_calls，图片消息挪到 tool 之后：

```python
# ui.py:do_response 中的顺序（tool 紧跟，图片殿后）
tool_msg = pack_msg("tool", "tool", tool_result, tool_call)
msgs = [tool_msg] + ([extra_msg] if extra_msg else [])
response = await self.brain.get_llm_response(msgs)
```

```python
# brain.py:166-169
response_msg = {"role": resp_msg.role, "content": resp_msg.content}
if resp_msg.tool_calls:
    # tool_calls 必须保留，否则后续 tool 角色消息找不到对应调用，API 返回 400
    response_msg["tool_calls"] = [tc.model_dump() for tc in resp_msg.tool_calls]
```

### 4.3 消息队列与合并

**问题**：LLM 处理中时新语音直接丢弃，连说几句话只剩一句。

**思路**：忙碌时也入队（上限 20 条防无限积压），消费者每次处理前把积压一次性取出按行合并——
连续语音碎片合成一条给 LLM，不逐条刷屏。

### 4.4 人设提示词

`SYSTEM_PROMPT` 进 config 可配，内置糯糯默认人设。要点是针对**气泡这个展示载体**做约束：
软萌口语、一两句话、禁用 markdown/列表（长 markdown 在 280px 气泡里没法看）。

## 5. 交互打磨

### 5.1 气泡 UI

`QLabel` 白底圆角、限宽 280px、自动换行，位于贴图上方；`QVBoxLayout` + `adjustSize()`
让窗口随气泡显隐伸缩。状态流：受理→"听到了，正在想…"（60s 兜底）→ 回复（10s 自动隐藏）→ 跳过/出错立即收起。

### 5.2 单图程序动画

只有一张贴图，用**整窗微动**模拟状态（不动贴图本身，动窗口位置，绕开 layout 冲突）：

```python
# ui.py:_anim_tick
self._anim_phase += 0.12
if self._anim_state == "thinking":
    offset = QPoint(round(3 * math.sin(self._anim_phase * 4)), 0)  # 快速晃动
elif self._anim_state == "talking":
    offset = QPoint(0, -abs(round(4 * math.sin(self._anim_phase * 2))))  # 弹跳
else:
    offset = QPoint(0, round(2 * math.sin(self._anim_phase)))  # 缓慢呼吸
self.move(self._base_pos + offset)
```

拖动时暂停动画（不跟用户抢窗口），松手后基准位置跟随。

### 5.3 单击输入 vs 拖动

单击桌宠唤起 `QLineEdit` 文字输入（回车进消息队列，Esc 收起），与拖动通过 **6px 位移阈值**区分；
双击"触碰"彩蛋保留，用标志位防双击误触输入框。

### 5.4 糯糯替换与 GIF

形象从胡桃换成糯糯（Q版弗洛洛），`Hutao` 类泛化为 `DeskFriend`。贴图路径进 config（`SPRITE`），
GIF 用 `QMovie` 播放、按 150 宽等比缩放：

```python
# ui.py:_load_sprite
if sprite_path.suffix.lower() == ".gif":
    self.movie = QMovie(str(sprite_path))
    self.movie.jumpToFrame(0)  # 先取一帧拿到原始尺寸，按比例算缩放
    frame_size = self.movie.currentImage().size()
    scaled = QSize(SPRITE_WIDTH, round(SPRITE_WIDTH * frame_size.height() / frame_size.width()))
    self.movie.setScaledSize(scaled)
```

## 6. 记忆系统（Phase 5/6）

桌宠从"健忘"到"记得你"：SQLite 持久化 + 滚动摘要 + 事实型长期记忆。
设计红线：**记忆故障绝不阻断对话**——摘要、抽取、合并三条 LLM 链路全部吞异常只打印。

### 6.1 MemoryStore：四张表

**思路**：`pet.db` 单文件 SQLite。`messages` 存完整历史（按 `turn_id` 归组一轮对话），
`summaries` 是恒为单行（`CHECK (id = 1)`）的滚动摘要，`facts` 存长期事实，`meta` 存抽取游标。
核心不变式：**未压缩消息（`summarized=0`）与内存 `context` 一一对应、顺序一致**——
启动时 `load_unsummarized()` 直接恢复上下文。

```python
# memory.py:55-67
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    msg_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    summarized INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    content TEXT NOT NULL,
    updated_at REAL NOT NULL
);
```

落库前过 `sanitize_message`：base64 图片替换成 `[image omitted]` 占位符，防止数据库被截图撑爆
（内存里的原消息不受影响）：

```python
# memory.py:14-22
def sanitize_message(msg):
    """返回消息的落库副本:把 base64 图片等大体积内容替换为占位符,避免 pet.db 膨胀。"""
    m = copy.deepcopy(msg)
    content = m.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part["image_url"] = {"url": "[image omitted]"}
    return m
```

### 6.2 滚动摘要压缩

**思路**：未压缩轮次超 30 轮触发；最老的轮次（保留最近 10 轮不动）交给 LLM，
**已有摘要 + 新对话 → 输出更新后的完整摘要**（200 字内滚动更新），然后标记旧轮已压缩、同步裁内存：

```python
# brain.py:181-200
summary_context = [
    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
    {
        "role": "user",
        "content": (
            f"已有摘要：\n{self.summary or '（无）'}\n\n"
            f"需要并入的新对话：\n{render_messages(old_msgs)}\n\n"
            "请输出更新后的完整摘要。"
        ),
    },
]
response = await self.get_response_with_context(summary_context)
new_summary = (response.choices[0].message.content or "").strip()
if not new_summary:
    return
self.memory.set_summary(new_summary)
self.memory.mark_summarized(turn_ids)
self.summary = new_summary
del self.context[: len(old_msgs)]
```

### 6.3 memorize()：背景谈话只记不答

**思路**：`should_reply` 判否的消息不再直接扔掉——以 `[背景谈话，无需回应]` 标注独立成轮落库，
不生成回复，但对后续对话和事实抽取可见。桌宠由此能"听到"你跟别人的谈话并在合适时用上。

```python
# brain.py:130-137
def memorize(self, message):
    """判断为无需回复的听觉消息：只落记忆、不生成回复。"""
    self.turn_id = self.memory.next_turn_id()
    self._store({"role": "user", "content": f"[背景谈话，无需回应] {message}"})
```

### 6.4 facts：事实型长期记忆

**思路**：摘要会丢细节，重要事实（学校、偏好、计划、约定）需要结构化专表。
每次回复完成后批量处理：用 `meta` 里的游标取自上次抽取以来的全部轮次（**含背景谈话**，
背景消息攒批处理，避免每条都花一次 LLM 调用），连同现有 facts 发给 LLM，
输出纯 JSON 操作列表：

```python
# 输出格式（brain.py:58-60）
{
    "operations": [
        {"op": "add", "content": "..."},
        {"op": "update", "id": 1, "content": "..."},
        {"op": "delete", "id": 2},
    ]
}
```

**LLM 输出不可信，必须校验落库**：add 要求非空 content，update/delete 要求 id 真实存在，其余一律忽略：

```python
# brain.py:236-247
def _apply_fact_ops(self, ops):
    existing_ids = {fid for fid, _ in self.memory.get_facts()}
    for op in ops:
        action = op.get("op")
        content = op.get("content")
        fact_id = op.get("id")
        if action == "add" and isinstance(content, str) and content.strip():
            self.memory.add_fact(content.strip())
        elif action == "update" and fact_id in existing_ids and isinstance(content, str) and content.strip():
            self.memory.update_fact(fact_id, content.strip())
        elif action == "delete" and fact_id in existing_ids:
            self.memory.delete_fact(fact_id)
```

facts 超 50 条时再让 LLM 合并压缩到 30 条（解析失败原表不动）。
注入方式：全部 facts 编号后作为第二条 system 消息全量注入，摘要第三条：

```python
# brain.py:121-126
facts = self.memory.get_facts()
if facts:
    lines = "\n".join(f"{i}. {content}" for i, (_, content) in enumerate(facts, 1))
    messages.append({"role": "system", "content": f"以下是你记住的关于用户的事情：\n{lines}"})
if self.summary:
    messages.append({"role": "system", "content": f"以下是你和用户此前对话的摘要：\n{self.summary}"})
```

## 7. 番外：UESTC_DF-rs

打包分发阶段重新评估了 Rust：Python 版的包袱是 300~500MB 运行时 + 打不进去的
gi/GStreamer/libportaudio 系统依赖 + 首启三座大山（装包、下模型、点授权）。
Rust 目标形态是 **30~50MB 单二进制 + 模型文件，下载即用**（egui + tokio + cpal + ort +
whisper.cpp + zbus/pipewire，零额外系统依赖）。

已独立建仓 `UESTC_DF-rs`（`Develop/rust/projects/`），开发文档包含总设计与 M0~M4 验收标准，
本文记录的 9 条踩坑经验全部迁移为设计约束（VAD 64 点上下文、tool 消息顺序、PipeWire 省电等）。
当前状态：M0（GUI 骨架）待开工。

## 附：经验清单（ distilled ）

1. Wayland 下截屏的唯一正解是 ScreenCast Portal + persist token；其余路径要么被拒要么弹窗要么闪光
2. 官方 SDK 包装器里藏着模型契约（如 VAD v5 的上下文拼接），绕过包装器前先读它的源码
3. "已缓存"不等于"离线可用"——HF 系库默认每次都联网校验，`local_files_only` 是好朋友
4. LLM 的结构化输出永远要校验落库；tool calling 的消息序列约束（tool 紧跟 tool_calls）比想象中严格
5. 判定类 LLM 调用别带 tools，否则模型会用行动代替回答
6. 桌面应用的 Ctrl+C 要靠 `SIG_DFL`，Qt 事件循环不会把执行权还给 Python
7. 回退策略要考虑用户体验：宁可功能暂时缺席（跳过截图），不可体验降级（闪光）
