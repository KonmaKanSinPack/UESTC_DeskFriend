"""astrbot 回复后端：桌宠的"大脑"交给 AstrBot 的桃桃。

- 对话：OneBot 11 伪装通道（onebot_bridge）——发消息事件 → 桃桃回复
- 判定：本地唤醒规则（不调 LLM，省 token），替代原来的 LLM should_reply
- 记忆：三接口空操作（记忆由 AstrBot 全局记忆接管）
- 屏幕感知：状态机（idle 60s / active 20s 观察）+ 变化门控 + 冷却，门控通过后经
  observe_sink 把"主动观察"文案交主控入统一消息队列（与用户消息同一处理/呈现路径，
  2026-09-03 起替代直发桥+冒泡旁路）；回复"无"类返回空 content=完全静默，
  有内容时仅活跃期朗读（speak 标志）
- 工具："LLM 主动看屏幕"用文本指令协议——桃桃回复含 `[look_at_screen]` 标记时，
  桌宠本地截屏附图追问（≤3 轮），最终回复不含标记
"""

import asyncio
import re

from onebot_bridge import OneBotBridge, current_loop

from .base import BackendResponse, ReplyBackend
from .judger import _extract_judge_text

# 屏幕关键词：消息文本命中即本地截屏，附图一起发给桃桃（桃桃多模态看图）
SCREEN_KEYWORDS = ("屏幕", "截图", "看看", "桌面")

# [look_at_screen] 协议：桃桃回复里出现该标记 → 桌宠截屏附图追问
LOOK_AT_SCREEN_MARKER = "[look_at_screen]"
LOOK_AT_SCREEN_MAX_ROUNDS = 3

# 对话后保持"活跃观察"的时长（active 态观察周期更短）
ACTIVE_AFTER_CHAT = 120  # 秒

# 主动观察回复中的"无话可说"集合（命中则静默丢弃，不冒泡）
NO_REPLY_SET = {"无", "没有", "不用", "不需要", "没什么", "没啥", "没事"}

# 主动观察消息的来源标记：spine 据此入 proactive 源、get_llm_response 据此走自发分支
# （不更新用户时间窗/不消费打断标记/回复按 _is_no_reply 判空/speak 按活跃期判定）
PROACTIVE_MARKER = "（桌宠主动观察）"

# 主动观察消息文案：明确告诉桃桃可以只回「无」；含"屏幕/截图"关键词，
# 走 get_llm_response 时 SCREEN_KEYWORDS 路径会自动截屏附图
PROACTIVE_OBSERVE_TEXT = (
    PROACTIVE_MARKER + "屏幕内容有新变化，请看截图。"
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
        self.observe_sink = None  # spine 挂的回调：门控通过后把观察文案交主控入统一队列（brain 门面转发）
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

        - 文本命中屏幕关键词 → 本地截屏附图（主动观察文案自带关键词，同路复用）
        - 回复含 [look_at_screen] → 截屏附图追问（≤3 轮），返回最终回复
        - PROACTIVE_MARKER 前缀的自发消息：不更新用户时间窗、不消费打断标记，
          "无"类回复返回空 content（主控完全静默），speak 按活跃期时间窗判定
        """
        loop = current_loop()
        segments = self._msg_to_segments(message)
        text = "".join(s["data"].get("text", "") for s in segments if s.get("type") == "text")
        proactive = text.startswith(PROACTIVE_MARKER)
        if not proactive:
            # 用户交互时间窗（活跃观察期/用户静默门控的数据源）：自发消息不得重置，
            # 否则"用户静默 SCREEN_USER_SILENCE"观察门控会被自己的观察续命而失灵
            self._active_until = loop.time() + ACTIVE_AFTER_CHAT
            self._last_user_at = loop.time()

            # 打断标记注入：用户上次朗读被打断时，让桃桃知道说到哪了（用后清除）。
            # 仅用户消息可消费；自发消息不应拿走留给下一条用户消息的打断位置。
            if self.interruption and isinstance(message, str):
                prefix = self.interruption
                self.interruption = None
                message = f"[对话被打断] 你刚才说到『{prefix}』处被打断了。用户现在说：{message}"
            segments = self._msg_to_segments(message)

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
        if proactive and _is_no_reply(reply):
            # 桃桃觉得没话说：空 content 是主控"完全静默"的约定信号
            # （用户消息永不返回空——下方兜底文案挡着）
            return BackendResponse(content="")
        if not content:
            # 桥超时/断线/空回复：兜底文案仅供气泡反馈，标记 answered=False，
            # 让主控不朗读它——否则被扬声器放出→麦克风拾回→回声自回复环
            return BackendResponse(content="（桃桃没有回应…）", answered=False)
        if proactive:
            # 仅活跃期朗读：用户 2 分钟内（ACTIVE_AFTER_CHAT）交互过才开口，
            # 空闲期主动观察只冒泡不发声（打扰门控；门在时间窗上，呈现代码单一路径）
            speak = loop.time() < self._active_until
            return BackendResponse(content=content, speak=speak)
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
        if self.observe_sink is not None:
            # 统一消息流（2026-09-03）：门控通过后只把观察文案交主控入队（源=proactive），
            # 不再直接发桥——截图由 get_llm_response 的 SCREEN_KEYWORDS 路径自动附图
            # （取图时刻稍晚于门控哈希时刻，属可接受漂移）；与用户消息同队列串行，
            # 并发互斥由队列天然保证
            self.observe_sink(PROACTIVE_OBSERVE_TEXT)
