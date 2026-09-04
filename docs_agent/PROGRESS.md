# PROGRESS — 跨会话状态锚点

> 每轮结束必更（RULES 门禁）。新会话开工读此文件掌握「做到哪、接下去」，细节看 `session/` 对应日志。

- 更新：2026-09-03

## 当前状态

- 分支 `tao_tao`；测试基线 **165 passed**，ruff 全绿
- 回复后端双通道可用（`config.toml` 的 `BACKEND` 切换）：
  - `astrbot`（默认）：OneBot 11 伪装通道接 AstrBot 桃桃，记忆/人格由 AstrBot 接管
  - `openai`：直连兼容接口 + 自建记忆；判定独立走 `JUDGE_URL`（本地小模型），不占主模型额度
- 交互链路完整：语音/文字/双击彩蛋 → 气泡 + TTS 朗读 → 句间窗口打断 → 回声门控
- 开发机：Windows（见根 `local.md`）；Wayland/PipeWire 为 Linux 历史决策，路径保留在 vision 回退链中

## 里程碑（详录见 session/ 与 dev-journey.md）

- Phase 0~6 完成：工程化（uv/ruff/拆分）、Wayland 兼容、气泡 UI、依赖瘦身、
  短期记忆+滚动摘要、事实型长期记忆
- 2026-08-13：ui.py → skin+spine 拆分（零行为变化）；打断/回声状态机加固（Tier1+2 共 10 条，
  头号修复：窗口期录音被误判污染整段丢弃）
- 2026-08-14：openai 后端判定通道独立化（对话 MiniMax + 判定本地 qwen，提示词身份路由）
- 2026-08-21：docs 重组为双层体系（docs_agent 权威层 + docs_human 人读层），RULES 通用化
- 2026-09-03：desktop-pet-dev skill 对比审查（差距清单见当轮日志）；「关得掉」轮——
  托盘（贴图缩略）+ 右键退出 + Qt.Tool + spine 优雅收摊编排
- 2026-09-03（第二轮）：**回复流程统一**——主动观察并入主消息流（队列 (source,text)
  分流、observe_sink 替代 reply_sink 旁路、空 content=无话静默、仅活跃期朗读 speak 门）
- 2026-09-03（第三轮）：**ASR 换引擎**——SenseVoice 本地（sherpa-onnx，中文精度↑、
  0.52s/6s 音频 vs whisper-cuda 1.75s）+ Whisper 回退 + 模型自动下载 + 采集三小修
  （pre-roll/裁尾/增益）
- 2026-09-03（第四轮）：**流畅性快赢**（调研驱动：ElevenLabs/arXiv/Qt 文档）——
  动画帧率分档（idle 8fps/thinking·talking 25fps）、VAD 静音 1.5s→1.0s、
  受理气泡显示转写原文（听对没一眼可见）
- 2026-09-03（第五轮）：**UI 美化**——圆角卡气泡（暖白+桃桃红+柔投影+淡入淡出）、
  生成中三点指示（set_pending 命令）、输入框聚焦样式；用户已定方向：**形象换 Live2D**
  （vs 3D 已建议 Live2D，技术选型 live2d-py vs QtWebEngine 留该轮）
- 2026-09-04：hotfix——重启后图片占位符毒化上下文致 API 500（revive_message 还原边界）；
  **config 三文件分层**（common/openai/astrbot + 记忆阈值 MEMORY_* 配置化 +
  load_config 迁出 skin + 工厂显式传 config）；含用户改名 render_messages 助手名→桃桃
- 2026-09-04（第二轮）：**PSD 直驱渲染器**——prepare_psd（拆 L/R+扩边+manifest）+
  PsdRenderer（眨眼/视线/口型/呼吸/发摆）+ skin 按 SPRITE 后缀装配；演示宠物离屏
  出帧验证通过。**等用户重发 seethrough PSD**（原件已不在本机）后跑 prepare 即换真桃桃

## 待办（下一步候选，优先级自上而下）

- [ ] **openai 流式管道轮**（用户当前即 openai 后端，价值最高）：LLM 流式→句界切分→
      逐句喂 TTS，首句出声提前 2~5s（行业标配；astrbot 侧 OneBot 无流式是协议天花板）
- [ ] 动画渲染改造轮：整窗 move → 窗口内自绘偏移（DWM 合成器友好路径，CPU/流畅双收）
- [ ] P1 小轮：QSettings 位置持久化 + 多屏 clamp（帧率分档已随第四轮完成）
- [ ] AEC 全双工轮：设计草案已评审（持久输出流逐帧 tee + far 永不断 + APM 一次对齐），
      待写正式 plan+spec 后开工
- [ ] 久坐提醒（走现成气泡链路）；双击即时反馈补丁；review 带读（站 2~9 暂停中）
- [ ] Phase 4 打包分发（PyInstaller）；Phase 7 FTS5 历史检索；technical.md 增量充实

## 真机冒烟清单（用户，按轮累积）

- [ ] ① openai 模式判定/对话双通道 + 朗读/打断/双击彩蛋
- [ ] ② 「关得掉」轮——托盘图标为贴图缩略、任务栏与 Alt-Tab 无桌宠、托盘/右键
      两入口优雅退出（无残留进程）、朗读中退出即停嘴、连点不炸
- [ ] ③ 统一流——空闲期主动观察只冒泡、活跃期会朗读、无话完全静默、主动观察
      不与用户消息合并/互相打断、打断标记不被自发消息消费
- [ ] ④ ASR——日常对话识别精度对比（vs 旧 whisper）、首字完整、幻觉消失、
      冷启动下载模型一次后离线秒起、TTS 播放期间回声行为不变
- [ ] ⑤ 流畅性——idle 时任务管理器 CPU 显著下降、说话停顿 1s 不再截断丢字、
      气泡先见转写原文再见回复、呼吸动画节奏与之前一致
- [ ] ⑥ UI 美化——气泡圆角暖白+柔投影、淡入淡出不生硬、生成中三点跳动、
      回复覆盖无闪烁、输入框聚焦桃红描边、超时 10s 淡出自隐
- [ ] ⑦ PSD 渲染——（换真桃桃层后）眨眼随机左右独立、视线跟鼠标、说话嘴动、
      呼吸起伏、发丝轻摆、thinking 摆动、拖动正常、透明底；演示层可先行：
      SPRITE 指 assets/live2d/demo/manifest.json

## 已知风险 / 注意

- torch DLL 偶发 `WinError 1114`（Windows 环境问题，重试即过，非代码问题）
- CosyVoice2 本地 TTS **未完成**（依赖栈敏感，合成乱码，见 DEVELOPMENT §8.1）；生产用 SiliconFlow
- assets/persona/taotao.md 为用户私人文件，**不入库**（保持 untracked）
