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

# 消息来源标记：队列元素 (source, text) 的 source 值。
# user = 耳/皮/彩蛋等用户输入；proactive = astrbot 后端主动观察（自发，
# 2026-09-03 起入统一消息流：与用户消息同一队列/同一呈现路径，仅分流不合并）
USER_SOURCE = "user"
PROACTIVE_SOURCE = "proactive"


def group_pending(pending):
    """积压分流：连续的 user 源合并成一条（语音碎片语义），proactive 源独立成条。

    防污染：主动观察是桌宠自发内容，与用户话音 "\\n" 拼接会让桃桃困惑；
    分流后两者在同一队列串行（天然互斥，替代旧的 bridge.busy 防打架）但互不合并。
    纯函数便于单测。
    """
    grouped = []
    for source, text in pending:
        if source == USER_SOURCE and grouped and grouped[-1][0] == USER_SOURCE:
            grouped[-1] = (USER_SOURCE, grouped[-1][1] + "\n" + text)
        else:
            grouped.append((source, text))
    return grouped


class Spine:
    """主控中枢：器官装配 + 消息队列 + 判定 + 回复编排 + 打断编排。

    全部依赖可注入（测试用）；生产路径默认装配（照 Brain(backend=None)/Mouth(tts=None) 模式）。
    """

    def __init__(self, face=None, config=None, brain=None, vision=None, listen=None, mouth=None, quit_app=None):
        self.face = face or Skin()
        config = config or load_config()
        self.brain = brain or Brain()
        self.vision = vision or Vision()
        self.listen = listen or Listen()
        self.mouth = mouth or Mouth(config)
        # 退出回调由装配层（main）注入 app.quit——spine 零 Qt 依赖的红线靠它保住
        self._quit_app = quit_app
        self._shutting_down = False  # 幂等闩：托盘+右键并发/菜单双击不重复收摊

        # —— 接线（"中枢神经"的本体）：器官信号 → 主控处理 ——
        self.listen.mouth = self.mouth  # 耳朵引用嘴：朗读期间忽略扬声器回声（单向只读）
        # 插嘴打断：句间监听窗口内检测到用户说话 → 打断（复用现有注入链路）
        self.listen.interrupt_requested.connect(self._interrupt_tts)
        self.listen.text_signal.connect(self.on_heard_text)  # 发射器.信号.connect(接收器)
        self.face.text_submitted.connect(self._on_text_submitted)
        self.face.touched.connect(self._on_touched)
        self.face.quit_requested.connect(self._on_quit_requested)  # 托盘/右键「退出」
        # 主动观察提交（astrbot 屏幕感知门控通过）：proactive 源入统一队列，
        # 响应与用户消息走同一条 do_response（2026-09-03 起替代 reply_sink 冒泡旁路）
        self.brain.observe_sink = self._on_proactive_observe

        # 自锁
        self.is_busy = False
        # 消息队列-》使用生产者-消费者结构实现
        # 带上限：积压超过 20 条时丢弃新消息，防止无限积压
        self.message_queue = asyncio.Queue(maxsize=20)

    def start(self):
        """起消费者任务（与构造分离：test_spine 无需活 loop 即可实例化）。"""
        asyncio.get_event_loop().create_task(self.on_received_message_consumer())

    # ---------- 退出编排 ----------

    def _on_quit_requested(self):
        """皮信号入口（托盘/右键「退出」）：把收摊协程挂到 loop（同 on_heard_text 模式）。

        不能在信号槽里直接 await——Qt 信号槽是同步调用栈，这里只负责把协程
        create_task 交给 qasync 主 loop，真正的收摊在协程里异步进行。
        """
        asyncio.get_event_loop().create_task(self._shutdown())

    async def _shutdown(self):
        """按资源依赖序收摊：脑 → 嘴 → 退出事件循环。幂等：重复请求直接忽略。

        为什么是这个顺序（反了会怎样）：
        1. brain.stop() 必须最先——OneBot 桥要结算在飞请求（等回复中的对话、屏幕
           观察循环取消）。若先退出 Qt 循环，qasync loop 一关这些协程直接蒸发，
           AstrBot 侧留下半截会话。
        2. mouth.stop() 其次——打断在播朗读（否则声音会一直播到进程真正死亡），
           并给 TTS 一次释放资源的机会。
        3. quit_app() 最后——它是 main 注入的 app.quit 回调：spine 若直接 import
           QApplication 就破坏「主控零 Qt」红线（单测也得拉起 Qt），所以退出循环
           这个动作交回装配层执行；app.quit 后 main 的 `with loop:` 正常关闭。
        复现：新增器官需要在退出时收尾的，按依赖序在本方法追加一行 stop。
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        await self.brain.stop()
        await self.mouth.stop()
        if self._quit_app:
            self._quit_app()

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

        if not response.content:
            # 主动观察"无话可说"（astrbot 后端对"无"类回复返回空串）：完全静默——
            # 不气泡不朗读。用户消息永不返回空（astrbot 有兜底文案/openai 正常有内容）
            return

        print(response.content)
        self.face.set_anim_state("talking")
        self.face.show_bubble(response.content)
        # 桥超时/断线兜底（answered=False）：气泡给反馈，但不朗读——否则兜底文案被
        # 扬声器放出→麦克风拾回→回声自回复环（详见 docs_agent/session/2026-08-13.md 组B）
        if not response.answered:
            return
        # 朗读回复（异步不阻塞对话队列；门控置位/尾巴释放由 Mouth 内部消化）。
        # speak=False：主动观察空闲期（用户 2 分钟内未交互）只冒泡不朗读——
        # 打扰门控由后端时间窗判定（BackendResponse.speak），呈现代码单一路径只多这一个条件
        if response.speak:
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
            self.message_queue.put_nowait((USER_SOURCE, text))  # 忙碌时也入队，等消费者空闲后处理
            self.face.show_bubble("听到了，正在想…", timeout_ms=60000)
            self.face.set_anim_state("thinking")
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_text_submitted(self, text):
        """皮信号入口：打字 → 打断朗读 + 入队（编排全部在脊柱，皮只广播）。"""
        print(f"接收到文字消息：{text}")
        self._interrupt_tts()  # 打断朗读，记录打断位置
        try:
            self.message_queue.put_nowait((USER_SOURCE, text))
            self.face.show_bubble("听到了，正在想…", timeout_ms=60000)
            self.face.set_anim_state("thinking")
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_touched(self):
        """皮信号入口：双击"触碰"彩蛋 → 入队。"""
        print("接收到鼠标触碰事件")
        try:
            self.message_queue.put_nowait((USER_SOURCE, "用户用鼠标触碰了你"))
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这条消息。")

    def _on_proactive_observe(self, text):
        """observe_sink 入口（astrbot 屏幕感知门控通过）：proactive 源入统一队列。

        与用户消息同一队列串行（天然互斥）；受理不打"正在想"气泡——自发观察
        在桃桃给出值得说的话之前对用户零打扰。
        """
        print("接收到主动观察提交")
        try:
            self.message_queue.put_nowait((PROACTIVE_SOURCE, text))
        except asyncio.QueueFull:
            print("消息队列已满，丢弃这次主动观察。")

    # ---------- 消费者（生产者-消费者结构） ----------

    async def on_received_message_consumer(self):
        while True:
            pending = [await self.message_queue.get()]  # 队列空时停留在这行
            while not self.message_queue.empty():
                pending.append(self.message_queue.get_nowait())
            # 分流后逐条处理：连续 user 合并（语音碎片），proactive 独立成条
            for source, message in group_pending(pending):
                self.is_busy = True
                try:
                    print(f"正在处理消息（{source}）：{message}")
                    # proactive 源跳过判定器：要不要观察已由后端四重门控决策，
                    # 判定器（should_reply）回答的是"用户这句话值不值得回"，语义不同
                    if source == PROACTIVE_SOURCE or await self.should_reply(message):
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
