"""DeskFriend 桌宠适配插件。

功能(与桌宠项目 UESTC_DeskFriend 配套,2026-09-09):
1. 桌宠来源消息(按 OneBot self_id 识别)追加输出风格约束:简短聊天、禁动作描写、
   禁 markdown——回复会进桌面气泡并被语音朗读。QQ/NapCat 等其它来源零侵入。
2. 注册 look_at_screen 工具:经 OneBot action("deskfriend_tool")调用桌宠侧的
   统一工具链执行,截图以 data-url 在**同一轮**追加进 agent 消息(mark_as_temp
   不入持久历史),桃桃看图作答。

API 事实核对自 AstrBot v4.27.2 源码(本机 D:/Astrbot/AstrBot):
- @filter.on_llm_request(event, req) 可改 req.system_prompt
- FunctionTool.call(context: ContextWrapper, **kwargs);context.messages 是
  agent 轮内消息列表(runner 自动维护)——同轮注图的官方数据面
- ImageURLPart(image_url=ImageURL(url="data:image/png;base64,..."))
- adapter.bot.call_action(action, ..., self_id=...) 支持任意 action + echo 回包
"""

from astrbot.api import FunctionTool, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from pydantic.dataclasses import dataclass

try:
    # 内部消息结构(暂无公开 API 面):升级 AstrBot 若变动此处,插件的"同轮注图"
    # 会退化——ImportError 分支里工具仍可用,只是截图不进对话(返回文本提示)
    from astrbot.core.agent.message import ImageURLPart, Message

    _IMAGE_INJECTION_OK = True
except ImportError:  # pragma: no cover
    _IMAGE_INJECTION_OK = False


@dataclass
class LookAtScreenTool(FunctionTool):
    """查看桌宠所在电脑的当前屏幕:桌宠侧执行截屏,截图直接进入本轮对话。"""

    plugin: object = None  # 回持插件实例(拿配置与适配器);置尾避免与基类字段序冲突

    name: str = "look_at_screen"
    description: str = "查看桌宠所在电脑(桌面宠物端)的当前屏幕内容,获取用户正在看的画面截图"
    parameters: dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {"type": "object", "properties": {}}

    async def call(self, context, **kwargs) -> str:
        return await self.plugin.call_pet_tool(context, "look_at_screen", kwargs)


class DeskFriendPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)
        cfg = getattr(self, "config", None) or {}
        self.pet_self_id = str(cfg.get("pet_self_id", "10001"))
        self.platform_name = cfg.get("aiocqhttp_platform_name", "aiocqhttp")
        self.style_addon = cfg.get("style_addon", "")
        self.enable_style = bool(cfg.get("enable_style_injection", True))
        self.enable_tools = bool(cfg.get("enable_tools", True))

        if self.enable_tools:
            # 工具全局注册(所有会话可用):QQ 侧的桃桃也能看家里电脑的屏——
            # 两个都是用户自己的,视为 feature;介意可在 AstrBot 里停用本插件
            self.context.add_llm_tools(LookAtScreenTool(plugin=self))
            logger.info("DeskFriend 工具已注册:look_at_screen")

    async def terminate(self):
        pass

    # ---------- 桌宠来源识别 + 风格注入 ----------

    def is_from_pet(self, event: AstrMessageEvent) -> bool:
        """按 OneBot self_id 判定消息是否来自桌宠连接。

        不用 sender user_id 判:桌宠的 USER_ID 就是主人本人 QQ,真 QQ 聊天会撞车;
        self_id(桌宠伪装的 10001)与 NapCat 的真实 bot QQ 天然可分,即使共用
        同一个 aiocqhttp 适配器。
        """
        try:
            return str(getattr(event.message_obj, "self_id", "")) == self.pet_self_id
        except Exception:
            return False

    @filter.on_llm_request()
    async def pet_style_and_tools_context(self, event: AstrMessageEvent, req: ProviderRequest):
        """桌宠来源:追加风格约束(追加不替换,人设在 AstrBot 人格里不动)。"""
        if not self.is_from_pet(event):
            return  # 其它来源零侵入
        if self.enable_style and self.style_addon:
            req.system_prompt = (req.system_prompt or "") + self.style_addon

    # ---------- 远程工具(OneBot action → 桌宠统一工具链) ----------

    def _find_aiocqhttp_adapter(self):
        for platform in self.context.platform_manager.platform_insts:
            try:
                if platform.meta().name == self.platform_name:
                    return platform
            except Exception:
                continue
        return None

    async def call_pet_tool(self, run_context, tool: str, args: dict) -> str:
        """经 OneBot action 调桌宠侧工具;截图 data-url 同轮注入 agent 消息。

        通道:aiocqhttp 适配器的 call_action(按 self_id 路由到桌宠连接,等 echo
        回包)。执行永远在桌宠侧的统一工具链(spine.execute_tool),本插件只传话。
        """
        adapter = self._find_aiocqhttp_adapter()
        if adapter is None:
            return f"桌宠工具不可用:找不到平台实例 {self.platform_name!r}"
        try:
            ret = await adapter.bot.call_action(
                action="deskfriend_tool",
                tool=tool,
                args=args or {},
                self_id=int(self.pet_self_id),
            )
        except Exception as e:
            return f"桌宠工具调用失败:{e}"

        if not isinstance(ret, dict) or ret.get("status") != "ok":
            why = ret.get("text", "未知原因") if isinstance(ret, dict) else repr(ret)
            return f"桌宠工具执行失败:{why}"

        text = str(ret.get("text", ""))
        image = ret.get("image")
        if image and _IMAGE_INJECTION_OK:
            # 同轮注图:直接把 data-url 追加为一条临时用户消息(mark_as_temp =
            # 只喂给 provider,不落持久历史——截图不该膨胀会话记录),下一个
            # LLM 调用在同一 agent 循环里就能看到
            try:
                run_context.messages.append(
                    Message(
                        role="user",
                        content=[ImageURLPart(image_url={"url": image}).mark_as_temp()],
                    )
                )
            except Exception as e:
                logger.warning(f"DeskFriend 截图注入失败(降级为文本):{e}")
                return f"{text}(截图未能加入对话)"
        return text or "已查看屏幕。"
