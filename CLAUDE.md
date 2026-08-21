# UESTC_DeskFriend 工作规范

## 设计原则（各司其职，详见 docs_agent/ARCHITECTURE.md）

- 大模型（大脑）只输出意图，绝不碰物理硬件；器官（listen/mouth/vision/skin）是纯物理层，
  绝不调用 LLM / 不依赖 brain；主控中心（spine，脊髓）监听器官信号、编排一切
- 写任何新代码前先按 docs_agent/ARCHITECTURE.md §四「代码归属判定」确定代码属于哪层
- 器官间只允许单向只读依赖（如耳读嘴的 speaking 门控）；事件走信号广播
- 评审/收尾时对照 ARCHITECTURE.md §七「评审检查清单」

## 工作方式

唯一权威是 **`docs_agent/RULES.md`**（开工前必读）：核心循环、开工清单、spec 规范、
收尾 6 项门禁、文档地图、修订流程。本文件不再复述其内容——双源维护必然漂移，
以 RULES 为准（其修订需与用户确认后记录在案）。
