# AGENTS — agent 自动加载入口

> 各类编码 agent 开工前从这里出发。

1. **工作方式唯一权威:`docs_agent/RULES.md`**——开工前必读。核心循环:
   读规则 → 读现状 → 出方案 brainstorm → 落 spec → 按 spec 实施 → 过 6 项门禁 → 等用户反馈
2. 写码前读 `docs_agent/ARCHITECTURE.md` §四「代码归属判定」(器官化硬规则:大脑不碰硬件、
   器官不懂大脑、主控编排一切)
3. 环境/命令/schema/常见坑:`docs_agent/DEVELOPMENT.md`。**本机差异与私有路径记根目录
   `local.md`**(gitignored,从 `docs_agent/local.template.md` 复制创建,拿到项目第一件事)
4. 跨会话状态:`docs_agent/PROGRESS.md` + `docs_agent/session/` 最近 1-2 篇日志
5. 人读层(非权威,内容不得与 docs_agent 冲突):`docs_human/design.md`(设计愿景)、
   `docs_human/overall.md`(全局技术文档)

注:根 `CLAUDE.md` 是 Claude Code 的等价入口,同样指向上述权威文件。
