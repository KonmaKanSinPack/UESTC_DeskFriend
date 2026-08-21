"""脊髓（spine）主控中枢：纯编排，不碰任何 widget。

职责：装配器官、监听器官信号（广播）→ should_reply 门卫 → 编排回复 → 命令器官。
器官不知道大模型存在，spine 不知道外观细节——各司其职（docs_agent/ARCHITECTURE.md）。

跨线程说明：PyQt5 信号对非 QObject 接收者同样经主线程事件循环投递（已用探针验证，
2026-08-13），所以 spine 的槽即使在耳线程 emit，也会在主线程执行——create_task
挂在 qasync 主 loop 上，与旧 @qasync.asyncSlot 等价，无需 QObject 基类。
"""

import asyncio

from backends.judger import JUDGE_SYSTEM_PROMPT
from brain import Brain, pack_msg, parse_tool_args
from listen import Listen
from mouth import Mouth
from skin import Skin, load_config
from vision import Vision


class Spine:
    """主控中枢：器官装配 + 消息队列 + 判定 + 回复编排 + 打断编排。

    全部依赖可注入（测试用）；生产路径默认装配（照 Brain(backend=None)/Mouth(tts=None) 模式）。
    """

    def __init__(self, face=None, config=None, brain=None, vision=None, listen=None, mouth=None):
        self.face = face or Skin()
        config = config or load_config()
        self.brain = brain or Brain()
        self.vision = vision or Vision()
        self.listen = listen or Listen()
        self.mouth = mouth or Mouth(config)

        # —— 接线（"中枢神经"的本体）：器官信号 → 主控处理 ——
        self.listen.mouth = self.mouth  # 耳朵引用嘴：朗读期间忽略扬声器回声（单向只读）
        # 插嘴打断：句间监听窗口内检测到用户说话 → 打断（复用现有注入链路）
        self.listen.interrupt_requested.connect(self._interrupt_tts)
        self.listen.text_signal.connect(self.on_heard_text)  # 发射器.信号.connect(接收器)
        self.face.text_submitted.connect(self._on_text_submitted)
        self.face.touched.connect(self._on_touched)
        # 主动冒泡回调：astrbot 后端（屏幕感知）发现值得说的话时直接显示气泡
        self.brain.reply_sink = self._on_proactive_bubble

        # 自锁
        self.is_busy = False
        # 消息队列-》使用生产者-消费者结构实现
        # 带上限：积压超过 20 条时丢弃新消息，防止无限积压
        self.message_queue = asyncio.Queue(maxsize=20)

    def start(self):
        """起消费者任务（与构造分离：test_spine 无需活 loop 即可实例化）。"""
        asyncio.get_event_loop().create_task(self.on_received_message_consumer())

    # ---------- 回复编排 ----------

    async def do_response(self, message):
        response = await self.brain.get_llm_response(message)
        while response.tool_calls:
            tool_call = response.tool_calls[0]
            print(f"接收到军师指令，准备运行: {tool_call.name}")
            tool_result, extra_msg = await self.tool_executer(tool_call)

            # 按照标准格式，把执行结果打包；tool 消息必须紧跟 assistant 的 tool_calls，
            # 截图等附加消息放在 tool 之后，否则 API 判定 role 序列非法返回 400
            tool_msg = pack_msg("tool", "tool", tool_result, tool_call)
            msgs = [tool_msg] + ([extra_msg] if extra_msg else [])

            # 第二次通信：带着结果回去要最终回复
            response = await self.brain.get_llm_response(msgs)

        print(response.content)
        self.face.set_anim_state("talking")
        self.face.show_bubble(response.content)
        # 桥超时/断线兜底（answered=False）：气泡给反馈，但不朗读——否则兜底文案被
        # 扬声器放出→麦克风拾回→回声自回复环（详见 docs_agent/session/2026-08-13.md 组B）
        if not response.answered:
            return
        # 朗读回复（异步不阻塞对话队列；门控置位/尾巴释放由 Mouth 内部消化）
        asyncio.get_event_loop().create_task(self.mouth.speak(response.content))
        # 回复已展示，再后台做记忆压缩（超限时把最老轮次并入摘要，失败不影响对话）
        await self.brain.maybe_compress()
        # 批量抽取自上次以来的事实（含此前攒下的背景谈话，失败不影响对话）
        await self.brain.maybe_extract_facts()

    async def tool_executer(self, tool_call):
        # tool_call 是统一 ToolCall 对象（openai 后端归一化而来；astrbot 后端不产生工具调用）
        func_name = tool_call.name
        _args_dict = parse_tool_args(tool_call.arguments)  # 目前工具都无参数，解析以备后续扩展

        if func_name == "look_at_screen":
            result = await self.vision.look_at_screen()
            if isinstance(result, dict):
                # 成功：图片消息由 do_response 按序插入 context（tool 消息之后）
                return "已查看屏幕并将图片信息加入上下文了哦", result
            # 失败/未就绪：返回的是提示文本，直接作为工具结果
            return result, None

    async def should_reply(self, message):
        try:
            # 判定提示词单一来源：backends/judger.py（两个后端共用，勿在此内联）
            judge_msg = pack_msg("system", "text", JUDGE_SYSTEM_PROMPT)
            user_msg = pack_msg("user", "text", f"用户的消息是：{message}")
            judge_context = [judge_msg, user_msg]

            response = await self.brain.get_response_with_context(judge_context)

            reply_decision = response.content.strip().lower()
            print(f"查看决策器纯净输出：{reply_decision}")
            return reply_decision == "true"
        except Exception as e:
            print(f"判断是否回复时出错了：{e}")
            return False

    # ---------- 打断编排 ----------

    def _interrupt_tts(self):
        """用户新消息到达：打断朗读并记录打断位置（异步不阻塞入队）。"""
        if self.mouth.busy:
            asyncio.get_event_loop().create_task(self._interrupt_tts_async())

    async def _interrupt_tts_async(self):
        # 主控中心编排：嘴只返回打断位置，注入对话上下文由这里决定
        prefix = await self.mouth.interrupt()
        if prefix:
            self.brain.set_interruption(prefix)
            print(f"朗读被打断，记录位置：…{prefix[-15:]}")

    # ---------- 消息入口（器官信号） ----------

    def on_heard_text(self, text):
        """耳信号入口：PyQt5 对非 QObject 接收者同样经主线程投递（探针已验证），
        在主线程把协程挂到 qasync loop（与旧 @qasync.asyncSlot 等价）。"""
        asyncio.get_event_loop().create_task(self._on_heard_text(text))

    async def _on_heard_text(self, text):
        print(f"接收到听觉消息：{text}")
        if self.mouth.is_echo(text):
            # 播放结束后余响（~1.3s）转写出的就是桃桃刚说的话：内容兜底，丢弃
            print(f"与刚朗读内容相似（疑似回声），丢弃：{text}")
            return
        self._interrupt_tts()  # 打断朗读，记录打断位置
        try:
            self.message_queue.put_nowait(text)  # 忙碌时也入队，等消费者空闲后处理
            self.face.show_bubble("听到了，正在想…", timeout_ms=60000)
            self.face.set_anim_state("thinking")
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_text_submitted(self, text):
        """皮信号入口：打字 → 打断朗读 + 入队（编排全部在脊柱，皮只广播）。"""
        print(f"接收到文字消息：{text}")
        self._interrupt_tts()  # 打断朗读，记录打断位置
        try:
            self.message_queue.put_nowait(text)
            self.face.show_bubble("听到了，正在想…", timeout_ms=60000)
            self.face.set_anim_state("thinking")
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_touched(self):
        """皮信号入口：双击"触碰"彩蛋 → 入队。"""
        print("接收到鼠标触碰事件")
        try:
            self.message_queue.put_nowait("用户用鼠标触碰了你")
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_proactive_bubble(self, text):
        """主动冒泡（astrbot 屏幕感知）：只显示气泡，不改动画状态。"""
        self.face.show_bubble(text)

    # ---------- 消费者（生产者-消费者结构） ----------

    async def on_received_message_consumer(self):
        while True:
            message = await self.message_queue.get()  # 当队列为空时就会永远停留在这一行
            # 把忙碌期间积压的消息一并取出合并（多为连续的语音片段）
            pending = [message]
            while not self.message_queue.empty():
                pending.append(self.message_queue.get_nowait())
            if len(pending) > 1:
                message = "\n".join(pending)
                print(f"合并了 {len(pending)} 条积压消息")
            self.is_busy = True
            try:
                print(f"正在处理消息：{message}")
                if await self.should_reply(message):
                    print("判断需要回复，正在处理消息...")
                    await self.do_response(message)
                else:
                    print("判断不需要回复，仅记入记忆。")
                    self.brain.memorize(message)  # 背景谈话只记不答
                    await self.brain.maybe_compress()  # 跳过回复的消息也要参与压缩
                    self.face.hide_bubble()
            except Exception as e:
                print(f"处理消息时出错了：{e}")
                self.face.hide_bubble()
            finally:
                self.is_busy = False
                # self.message_queue.task_done()  # 和 join 成对出现。当前队列没有调用 join，暂不启用。
