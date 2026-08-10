"""astrbot 回复后端：桌宠的"大脑"交给 AstrBot 的桃桃。

- 对话：OneBot 11 伪装通道（onebot_bridge）——发消息事件 → 桃桃回复
- 判定：本地唤醒规则（不调 LLM，省 token），替代原来的 LLM should_reply
- 记忆：三接口空操作（记忆由 AstrBot 全局记忆接管）
- 屏幕感知：状态机（idle 60s / active 20s 观察）+ 变化门控 + 冷却，
  屏幕显著变化时主动发桃桃"主动观察"消息，回复"无"类则静默丢弃，有内容则主动冒泡
- 工具："LLM 主动看屏幕"用文本指令协议——桃桃回复含 `[look_at_screen]` 标记时，
  桌宠本地截屏附图追问（≤3 轮），最终回复不含标记
"""

import asyncio
import re

from onebot_bridge import OneBotBridge, current_loop

from .base import BackendResponse, ReplyBackend

# 屏幕关键词：消息文本命中即本地截屏，附图一起发给桃桃（桃桃多模态看图）
SCREEN_KEYWORDS = ("屏幕", "截图", "看看", "桌面")

# [look_at_screen] 协议：桃桃回复里出现该标记 → 桌宠截屏附图追问
LOOK_AT_SCREEN_MARKER = "[look_at_screen]"
LOOK_AT_SCREEN_MAX_ROUNDS = 3

# 对话后保持"活跃观察"的时长（active 态观察周期更短）
ACTIVE_AFTER_CHAT = 120  # 秒

# 主动观察回复中的"无话可说"集合（命中则静默丢弃，不冒泡）
NO_REPLY_SET = {"无", "没有", "不用", "不需要", "没什么", "没啥", "没事"}

# 主动观察消息文案：明确告诉桃桃可以只回「无」
PROACTIVE_OBSERVE_TEXT = (
    "（桌宠主动观察）屏幕内容有新变化，请看截图。"
    "如果没有什么值得主动对主人说的话，只回复「无」；"
    "如果有值得说的（发现有趣的事、重要信息或需要提醒的事），就说出来。"
)


def clean_markdown(text):
    """剥掉桃桃回复里的轻量 markdown 符号，适配纯文本气泡（QLabel）显示。

    桃桃（AstrBot）默认用 markdown 回复（代码围栏、加粗、标题、行内代码），
    气泡是纯文本组件，直接显示会露出 ``` ** 等符号。清洗只动符号不动内容。
    """
    if not text:
        return text
    t = re.sub(r"(?m)^```[^\n]*\n?", "", text)  # 开头代码围栏行（含围栏内装饰）
    t = re.sub(r"\n?```\s*$", "", t)  # 结尾代码围栏行
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)  # **加粗**
    t = re.sub(r"\*([^*]+)\*", r"\1", t)  # *斜体*
    t = re.sub(r"`([^`]+)`", r"\1", t)  # `行内代码`
    t = re.sub(r"(?m)^#{1,6}\s*", "", t)  # # 标题
    return t.strip()


def _extract_judge_text(context):
    """从 ui.should_reply 构造的 judge context 提取用户消息原文。

    ui 固定用 "用户的消息是：{message}" 包装，剥离前缀后再交给决策器，
    否则纯语气词（"嗯嗯"）带着前缀会让判定失真。
    """
    text = ""
    if context:
        last = context[-1]
        content = last.get("content") if isinstance(last, dict) else ""
        if isinstance(content, str):
            text = content
            prefix = "用户的消息是："
            if text.startswith(prefix):
                text = text[len(prefix) :]
    return text


def phash(img, size=16):
    """感知哈希：缩放为 size×size 灰度图，每个像素与均值比较得 1 位 → size² 位整数。

    用于屏幕变化检测：哈希距离大 = 画面变化明显。
    """
    small = img.convert("L").resize((size, size))
    pixels = small.tobytes()  # L 模式每像素 1 字节（getdata 在 Pillow 14 弃用）
    avg = sum(pixels) / len(pixels)
    h = 0
    for i, v in enumerate(pixels):
        if v >= avg:
            h |= 1 << i
    return h


def phash_score(a, b, bits=256):
    """归一化汉明距离：0 完全相同，1 完全不同。"""
    return bin(a ^ b).count("1") / bits


def should_observe(change_score, now, last_proactive_at, cooldown, busy, last_user_at, user_silence, threshold):
    """主动观察门控（纯逻辑，便于单测）：全部满足才动用桃桃。

    - 画面变化幅度超阈值（变化门控，成熟方案共识：屏幕没变就不看）
    - 距上次主动观察超过冷却期（防打扰）
    - 当前不在对话中（不打断用户消息的等待）
    - 距上次用户输入超过静默期（用户正操作屏幕时不偷看）
    """
    if busy:
        return False
    if change_score < threshold:
        return False
    if now - last_proactive_at < cooldown:
        return False
    if now - last_user_at < user_silence:
        return False
    return True


def _is_no_reply(text):
    """主动观察的回复是否表示"无话可说"。"""
    t = (text or "").strip().strip("。！!？? ")
    return (not t) or (t in NO_REPLY_SET) or len(t) <= 1


class AstrBotBackend(ReplyBackend):
    name = "astrbot"

    def __init__(
        self,
        url,
        token="",
        user_id=1063310598,
        self_id=10001,
        nickname="老公",
        timeout=90.0,
        settle=2.0,
        screen_idle_interval=60,
        screen_active_interval=20,
        screen_change_threshold=0.05,
        screen_cooldown=180,
        screen_user_silence=120,
        bridge=None,
        judge=None,
        enable_observer=True,
    ):
        """bridge / judge 可注入（测试用）；enable_observer=False 关闭屏幕感知循环（测试用）。"""
        self.bridge = bridge or OneBotBridge(
            url=url,
            token=token,
            self_id=self_id,
            user_id=user_id,
            nickname=nickname,
            timeout=timeout,
            settle=settle,
        )
        self.screen_idle_interval = screen_idle_interval
        self.screen_active_interval = screen_active_interval
        self.screen_change_threshold = screen_change_threshold
        self.screen_cooldown = screen_cooldown
        self.screen_user_silence = screen_user_silence

        self.judge = judge  # LLM 决策器（should_reply 判定），由工厂注入
        self.reply_sink = None  # ui 挂的回调：主动冒泡显示（brain 门面转发）
        self.interruption = None  # TTS 朗读被打断的位置（brain.set_interruption 注入，用后清除）
        self._last_phash = None  # 上次观察到的屏幕哈希
        self._last_proactive_at = 0.0  # 上次主动观察时间
        self._last_user_at = 0.0  # 上次用户消息时间
        self._active_until = 0.0  # 活跃观察期截止时间（对话后一段时间）

        self.bridge.start()
        self._stop = False
        self._observe_task = None
        if enable_observer:
            self._observe_task = current_loop().create_task(self._observe_loop())

    async def stop(self):
        """停止观察循环与 bridge（幂等）：进程退出/测试收尾时调用。"""
        self._stop = True
        if self._observe_task is not None and not self._observe_task.done():
            self._observe_task.cancel()
            try:
                await self._observe_task
            except asyncio.CancelledError:
                pass  # 任务被取消属预期
        await self.bridge.stop()

    # ---------- 对话 ----------

    async def get_llm_response(self, message, model=None):
        """发消息给桃桃并等回复（超时/断线返回兜底文案）。

        - 文本命中屏幕关键词 → 本地截屏附图
        - 回复含 [look_at_screen] → 截屏附图追问（≤3 轮），返回最终回复
        """
        loop = current_loop()
        self._active_until = loop.time() + ACTIVE_AFTER_CHAT  # 对话后进入活跃观察
        self._last_user_at = loop.time()

        # 打断标记注入：用户上次朗读被打断时，让桃桃知道说到哪了（用后清除）。
        # 仅对话主通道（str 消息）注入；should_reply 判定不受影响。
        if self.interruption and isinstance(message, str):
            prefix = self.interruption
            self.interruption = None
            message = f"[对话被打断] 你刚才说到『{prefix}』处被打断了。用户现在说：{message}"

        segments = self._msg_to_segments(message)
        text = "".join(s["data"].get("text", "") for s in segments if s.get("type") == "text")
        if any(kw in text for kw in SCREEN_KEYWORDS):
            img = self._capture_segment()
            if img is not None:
                segments.append(img)
        if not segments:  # 防御：全是无法表达的消息段
            segments = [{"type": "text", "data": {"text": text or "…"}}]

        reply = ""
        for _ in range(1 + LOOK_AT_SCREEN_MAX_ROUNDS):
            reply = await self.bridge.send_message(segments)
            if LOOK_AT_SCREEN_MARKER not in reply:
                break
            img = self._capture_segment()  # 桃桃要看屏幕：截屏附图追问
            if img is None:
                break
            segments = [{"type": "text", "data": {"text": "（这是你刚才要看的屏幕）"}}, img]

        content = clean_markdown(reply.replace(LOOK_AT_SCREEN_MARKER, ""))
        if not content:
            content = "（桃桃没有回应…）"
        return BackendResponse(content=content)

    async def get_response_with_context(self, context, model=None, use_tools=False):
        """should_reply 判定：统一走 LLM 决策器（judge），不用本地关键词规则。

        判定在桌宠侧自调 LLM，不进 AstrBot 对话流，不污染桃桃记忆。
        """
        text = _extract_judge_text(context)
        if self.judge is None:
            print("判定器未配置（JUDGE 相关配置缺失），默认回复")
            return BackendResponse(content="true")
        reply = await self.judge.should_reply(text)
        return BackendResponse(content="true" if reply else "false")

    # ---------- 记忆（空操作：由 AstrBot 接管） ----------

    def memorize(self, message):
        """背景谈话：不落记忆、不生成回复（记忆由 AstrBot 全局记忆接管）。"""

    async def maybe_compress(self):
        """记忆压缩：空操作。"""

    async def maybe_extract_facts(self):
        """事实抽取：空操作。"""

    # ---------- 内部：消息 → OneBot segments ----------

    @staticmethod
    def _msg_to_segments(message):
        """str / dict / list[dict] → OneBot segments；tool 消息跳过（OneBot 无此概念）。"""
        msgs = message if isinstance(message, list) else [message]
        segments = []
        for m in msgs:
            if not isinstance(m, dict):
                content = m
            else:
                content = m.get("content")
                if m.get("role") == "tool":  # 工具结果：OneBot 通道无法表达，跳过
                    continue
            if isinstance(content, str):
                if content:
                    segments.append({"type": "text", "data": {"text": content}})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            segments.append(
                                {"type": "image", "data": {"file": f"base64://{url.split(',', 1)[-1]}", "subType": 0}}
                            )
        return segments

    def _capture_segment(self):
        """本地截屏 → OneBot image segment；截屏后端未就绪返回 None。"""
        from vision import capture_screen_to_image_url  # 惰性导入，避免 vision↔backends 循环依赖

        url = capture_screen_to_image_url()
        if url is None:
            return None
        return {"type": "image", "data": {"file": f"base64://{url.split(',', 1)[-1]}", "subType": 0}}

    # ---------- 屏幕感知状态机 ----------

    async def _observe_loop(self):
        """观察循环：idle/active 不同周期截屏，变化显著且门控通过则主动发桃桃观察。"""
        loop = asyncio.get_running_loop()
        while not self._stop:
            interval = self.screen_idle_interval
            if loop.time() < self._active_until:  # 对话活跃期观察更勤
                interval = self.screen_active_interval
            await asyncio.sleep(interval)
            if self._stop:
                break
            try:
                await self._check_and_observe(loop)
            except Exception as e:  # 主动观察失败绝不影响主循环
                print(f"主动观察失败：{e}")

    async def _check_and_observe(self, loop):
        from vision import grab_screenshot  # 惰性导入

        shot = grab_screenshot()
        if shot is None:
            return  # 截屏后端未就绪：跳过本次
        now = loop.time()
        h = phash(shot)
        change = phash_score(h, self._last_phash) if self._last_phash is not None else 1.0
        self._last_phash = h
        if not should_observe(
            change,
            now,
            self._last_proactive_at,
            self.screen_cooldown,
            self.bridge.busy,
            self._last_user_at,
            self.screen_user_silence,
            self.screen_change_threshold,
        ):
            return
        self._last_proactive_at = now
        await self._proactive_look()

    async def _proactive_look(self):
        """主动观察：发桃桃「屏幕变化 + 截图」；回复"无"类静默丢弃，有内容则冒泡。"""
        img = self._capture_segment()
        if img is None:
            return
        segments = [{"type": "text", "data": {"text": PROACTIVE_OBSERVE_TEXT}}, img]
        reply = await self.bridge.send_message(segments)
        if _is_no_reply(reply):
            return  # 桃桃觉得没话说：静默
        if self.reply_sink is not None:
            self.reply_sink(clean_markdown(reply))  # 桃桃主动冒泡
