# astrbot_plugin_deskfriend

[UESTC_DeskFriend](../README.md) 桌宠的 AstrBot 侧适配插件。

## 功能

1. **桌宠来源风格注入**:来自桌宠 OneBot 连接(按 `self_id` 识别)的消息,在 LLM
   请求的 system prompt 末尾追加输出约束——回复将进桌面气泡并被语音朗读,所以要求
   简短聊天化、**禁止动作/神态描写**(`*摸头*`、`(微笑)` 类)、禁 markdown。
   QQ/NapCat 等其它来源**零侵入**。
2. **look_at_screen 工具**:桃桃可 function-calling 调用,经 OneBot action
   (`deskfriend_tool`)转发到桌宠侧**统一工具链**执行(与桌宠 openai 后端同一条
   `spine.execute_tool`),截图以 data-url 在同一轮注入 agent 消息(临时消息,
   不落持久历史)。

## 安装(家庭 AstrBot)

把整个 `astrbot_plugin_deskfriend/` 目录拷到 AstrBot 的 `data/plugins/` 下,
WebUI 插件页重载;或 git clone 到该目录。

## 配置(_conf_schema.json,WebUI 可改)

| 键 | 默认 | 说明 |
|---|---|---|
| `pet_self_id` | `"10001"` | 桌宠伪装的 OneBot self_id,须与桌宠 `config/astrbot.toml` 的 `ASTRBOT_SELF_ID` 一致 |
| `aiocqhttp_platform_name` | `"aiocqhttp"` | 工具下发的适配器实例名 |
| `enable_style_injection` | `true` | 关掉则桌宠来源也不注入 |
| `style_addon` | 内置文案 | 追加的约束文案(可自定义) |
| `enable_tools` | `true` | 是否注册桌宠工具 |

## 依赖的桌宠侧协议(须配套)

桌宠 ≥ 2026-09-09 版本:`onebot_bridge` 支持 `deskfriend_tool` action
(`{tool, args}` → echo 回 `{status, text, image}`)。旧版桌宠收到该 action
只会回 ok+空数据,工具返回"执行失败"——升级桌宠即可。

## 已知边界

- 工具全局注册:QQ 侧桃桃也能调 `look_at_screen`(看家里电脑的屏);介意可停用
  `enable_tools` 或整插件
- 同轮注图依赖 AstrBot 内部消息结构(`astrbot.core.agent.message`),升级
  AstrBot 后若 ImportError,自动降级为"仅文本结果"(截图不进对话但工具不报错)
- 未在本机 AstrBot 运行时实测(家部署):首轮真机调试清单见桌宠仓库
  `docs_agent/session/2026-09-09.md`
