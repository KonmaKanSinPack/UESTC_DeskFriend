"""Spine 主控中枢编排逻辑单测：全假器官注入，无需 QApplication / 麦克风 / LLM。"""

import asyncio

from backends.base import BackendResponse
from spine import Spine


class StubSignal:
    """pyqtSignal 的最小替身：提供 connect/emit 面（普通类即可，Qt 零依赖）。"""

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class FakeFace:
    """外观替身：记录命令调用（show_bubble/hide_bubble/set_anim_state）。"""

    def __init__(self):
        self.text_submitted = StubSignal()
        self.touched = StubSignal()
        self.quit_requested = StubSignal()
        self.bubbles = []  # (text, timeout_ms)
        self.anim_states = []
        self.hide_count = 0

    def show_bubble(self, text, timeout_ms=10000):
        self.bubbles.append((text, timeout_ms))

    def set_anim_state(self, state):
        self.anim_states.append(state)

    def hide_bubble(self):
        self.hide_count += 1


class FakeListen:
    """耳替身：信号面 + mouth 引用槽。"""

    def __init__(self):
        self.text_signal = StubSignal()
        self.interrupt_requested = StubSignal()
        self.mouth = None


class FakeBrain:
    """脑替身：可编排判定结果/回复内容/异常。"""

    def __init__(self):
        self.reply_sink = None
        self.should_reply_result = "true"
        self.response = BackendResponse(content="你好呀")
        self.raise_on_reply = False
        self.replies = []
        self.memorized = []
        self.interruptions = []
        self.compress_count = 0
        self.extract_count = 0
        self.stop_count = 0

    async def stop(self):
        self.stop_count += 1

    async def get_llm_response(self, message, model=None):
        if self.raise_on_reply:
            raise RuntimeError("boom")
        self.replies.append(message)
        return self.response

    async def get_response_with_context(self, context, model=None, use_tools=False):
        return BackendResponse(content=self.should_reply_result)

    def memorize(self, message):
        self.memorized.append(message)

    async def maybe_compress(self):
        self.compress_count += 1

    async def maybe_extract_facts(self):
        self.extract_count += 1

    def set_interruption(self, prefix):
        self.interruptions.append(prefix)


class FakeMouth:
    """嘴替身：busy / is_echo / 打断 / 朗读探针。"""

    def __init__(self):
        self.busy = False
        self.interrupt_count = 0
        self.interrupt_prefix = "第一句。"
        self.echo_texts = set()
        self.speaks = []
        self.stop_count = 0

    def is_echo(self, text):
        return text in self.echo_texts

    async def speak(self, text):
        self.speaks.append(text)

    async def interrupt(self):
        self.interrupt_count += 1
        return self.interrupt_prefix

    async def stop(self):
        self.stop_count += 1


class FakeVision:
    async def look_at_screen(self):
        return "已查看屏幕并将图片信息加入上下文了哦", None


def make_spine(**overrides):
    """装配注入全假器官的 Spine；返回 (spine, face, brain, mouth)。"""
    kw = dict(face=FakeFace(), brain=FakeBrain(), vision=FakeVision(), listen=FakeListen(), mouth=FakeMouth())
    kw.update(overrides)
    spine = Spine(**kw)
    return spine, kw["face"], kw["brain"], kw["mouth"]


# ---------- 消费者 ----------


def test_consumer_merges_backlog_and_replies():
    """忙碌期间积压的消息一次性取出按行合并成一条给脑。"""

    async def scenario():
        spine, face, brain, mouth = make_spine()
        spine.start()
        await spine.message_queue.put("消息A")
        await spine.message_queue.put("消息B")
        await asyncio.sleep(0.05)
        assert brain.replies == ["消息A\n消息B"]
        assert ("你好呀", 10000) in face.bubbles
        assert "talking" in face.anim_states
        assert mouth.speaks == ["你好呀"]  # 回复被交给嘴朗读（异步不阻塞队列）

    asyncio.run(scenario())


def test_should_reply_false_memorizes_and_hides():
    """判定不回复：只落记忆 + 收气泡，不生成回复。"""

    async def scenario():
        brain = FakeBrain()
        brain.should_reply_result = "false"
        spine, face, _, _ = make_spine(brain=brain)
        spine.start()
        await spine.message_queue.put("背景谈话")
        await asyncio.sleep(0.05)
        assert brain.memorized == ["背景谈话"]
        assert face.hide_count == 1
        assert face.bubbles == []

    asyncio.run(scenario())


def test_answered_false_shows_bubble_without_speak():
    """组B 回归：兜底文案（answered=False）只显示气泡，不朗读/不压缩/不抽取。"""

    async def scenario():
        brain = FakeBrain()
        brain.response = BackendResponse(content="（桃桃没有回应…）", answered=False)
        spine, face, _, mouth = make_spine(brain=brain)
        spine.start()
        await spine.message_queue.put("你在吗")
        await asyncio.sleep(0.05)
        assert ("（桃桃没有回应…）", 10000) in face.bubbles
        assert mouth.speaks == []
        assert brain.compress_count == 0
        assert brain.extract_count == 0

    asyncio.run(scenario())


def test_do_response_error_releases_busy_and_hides():
    """处理异常：收气泡 + finally 释放 is_busy，不阻断后续消息。"""

    async def scenario():
        brain = FakeBrain()
        brain.raise_on_reply = True
        spine, face, _, _ = make_spine(brain=brain)
        spine.start()
        await spine.message_queue.put("触发异常")
        await asyncio.sleep(0.05)
        assert face.hide_count == 1
        assert spine.is_busy is False

    asyncio.run(scenario())


# ---------- 消息入口 ----------


def test_text_submitted_interrupts_and_queues():
    """打字：打断朗读 + 打断位置注入 + 入队 + 受理反馈。"""

    async def scenario():
        spine, face, brain, mouth = make_spine()
        mouth.busy = True  # 朗读中：打字应触发打断
        spine.start()
        spine._on_text_submitted("你好")
        assert spine.message_queue.qsize() == 1
        assert ("听到了，正在想…", 60000) in face.bubbles
        assert "thinking" in face.anim_states
        await asyncio.sleep(0.05)
        assert mouth.interrupt_count == 1
        assert brain.interruptions == ["第一句。"]
        assert brain.replies == ["你好"]

    asyncio.run(scenario())


def test_touched_queues_easter_egg():
    """双击触碰：皮广播 touched 信号 → 脊柱入队彩蛋消息（验证接线）。"""

    async def scenario():
        spine, face, brain, _ = make_spine()
        spine.start()
        face.touched.emit()
        await asyncio.sleep(0.05)
        assert brain.replies == ["用户用鼠标触碰了你"]

    asyncio.run(scenario())


def test_queue_full_drops_gracefully():
    """队列满（20 条）：丢弃新消息不抛异常。"""

    spine, _, _, _ = make_spine()  # 不起 start：无消费者，队列不被取走
    for i in range(20):
        spine.message_queue.put_nowait(f"m{i}")
    spine._on_text_submitted("溢出消息")
    assert spine.message_queue.qsize() == 20


def test_echo_text_dropped():
    """转写结果与刚朗读内容相似（is_echo）：直接丢弃，不入队不气泡。"""

    async def scenario():
        spine, face, _, mouth = make_spine()
        mouth.echo_texts.add("桃桃刚说的话")
        spine.start()
        spine.on_heard_text("桃桃刚说的话")
        await asyncio.sleep(0.05)
        assert spine.message_queue.empty()
        assert face.bubbles == []

    asyncio.run(scenario())


# ---------- 退出编排 ----------


def test_quit_requested_stops_organs_in_order_and_quits():
    """退出请求：脑先停、嘴再停、quit_app 最后且恰好一次（资源依赖序）。"""

    async def scenario():
        order = []
        brain, mouth = FakeBrain(), FakeMouth()
        brain.stop = _probed_stop(order, "brain", brain.stop_count)
        mouth.stop = _probed_stop(order, "mouth", mouth.stop_count)
        quits = []
        spine, *_ = make_spine(brain=brain, mouth=mouth, quit_app=lambda: quits.append(True))
        spine.face.quit_requested.emit()
        await asyncio.sleep(0.05)
        assert order == ["brain", "mouth"]
        assert quits == [True]

    asyncio.run(scenario())


def _probed_stop(order, name, _):
    """把替身的 stop 包一层顺序探针（忽略原计数，顺序为准）。"""

    async def stop():
        order.append(name)

    return stop


def test_quit_idempotent_on_repeated_requests():
    """幂等闩：托盘+右键并发/菜单双击的重复退出请求只收摊一次。"""

    async def scenario():
        spine, _, brain, mouth = make_spine(quit_app=lambda: None)
        spine.face.quit_requested.emit()
        spine.face.quit_requested.emit()
        await asyncio.sleep(0.05)
        assert brain.stop_count == 1
        assert mouth.stop_count == 1

    asyncio.run(scenario())


def test_quit_without_quit_app_is_safe():
    """未注入 quit_app（纯测试装配）：器官仍收摊，不抛异常。"""

    async def scenario():
        spine, _, brain, mouth = make_spine()  # 不传 quit_app
        spine._on_quit_requested()
        await asyncio.sleep(0.05)
        assert brain.stop_count == 1
        assert mouth.stop_count == 1

    asyncio.run(scenario())
