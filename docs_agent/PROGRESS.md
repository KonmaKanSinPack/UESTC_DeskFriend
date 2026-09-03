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

## 待办（下一步候选，优先级自上而下）

- [ ] 真机冒烟（用户）：① openai 模式判定/对话双通道 + 朗读/打断/双击彩蛋
      ② 「关得掉」轮——托盘图标出现且为贴图缩略、任务栏与 Alt-Tab 无桌宠、
      托盘/右键两入口均优雅退出（无残留进程）、朗读中退出即停嘴、连点不炸
      ③ 统一流——空闲期主动观察只冒泡、活跃期会朗读、无话完全静默、
      主动观察不与用户消息合并/不互相打断、打断标记不被自发消息消费
- [ ] 双击无即时反馈补丁（"听到了，正在想…"气泡 + thinking 动画）
- [ ] AEC 全双工轮：设计草案已评审（持久输出流逐帧 tee + far 永不断 + APM 一次对齐），
      待写成正式 plan+spec 后开工
- [ ] review 带读（站 2~9，暂停中，用户定时间）
- [ ] Phase 4 打包分发（PyInstaller）；Phase 7 FTS5 历史检索
- [ ] technical.md 骨架 → 接口变更时增量充实

## 已知风险 / 注意

- torch DLL 偶发 `WinError 1114`（Windows 环境问题，重试即过，非代码问题）
- CosyVoice2 本地 TTS **未完成**（依赖栈敏感，合成乱码，见 DEVELOPMENT §8.1）；生产用 SiliconFlow
- assets/persona/taotao.md 为用户私人文件，**不入库**（保持 untracked）
