"""OneBot 11 反向 WebSocket 伪装客户端（桌宠 = 一个迷你 NapCat）。

角色：桌宠伪装成 OneBot 实现，作为 WebSocket 客户端**反向连接** AstrBot 的
OneBot 适配器服务端，走标准 OneBot 11 协议：

- 上行：上报 `message` 事件（private，user_id 固定为老公），AstrBot 把它当
  QQ 私聊消息喂给对话流（记忆 + 人格）→ 桃桃回复
- 下行：接收 `action` 请求并响应——`send_private_msg`/`send_msg` 里的回复
  文本交给回复结算；`get_login_info` 返回伪装身份；其余回 ok 避免适配器报错重试

回复结算（静默窗口）：AstrBot 可能连发多条回复（如"回复中提示"占位 + 最终回复），
收到一条后等待 settle 秒，若期间无新回复则视为最终回复（取最后一条）。
"""

import asyncio
import json
import logging
import random
import time

from websockets.asyncio.client import connect

logger = logging.getLogger(__name__)

# OneBot 11 动作请求："echo" 字段原样回传用于对账
# 未知动作返回 ok + 空数据（比 retcode=1002 更能避免适配器重试刷屏）


def current_loop():
    """拿到当前线程的事件循环。

    优先返回**运行中**的 loop；没有运行中 loop 时回退到已设置（set_event_loop）的 loop。
    回退分支对应 qasync 的时序：main.py 在 `loop.run_forever()` 之前就构造 DeskFriend
    （内部创建 Brain/后台任务），此时 loop 已 set 但未运行，`get_running_loop()` 会抛
    RuntimeError（曾在真机启动时踩到）。
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.get_event_loop()


def extract_message_text(message):
    """从 OneBot message 字段提取纯文本：str 原样返回，segments 数组拼 text 段。"""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for seg in message:
            if isinstance(seg, dict) and seg.get("type") == "text":
                parts.append(str(seg.get("data", {}).get("text", "")))
        return "".join(parts)
    return ""


class OneBotBridge:
    """OneBot 11 反向 WS 客户端：常驻连接 + 心跳 + 消息上报 + 动作响应 + 回复结算。"""

    def __init__(self, url, token="", self_id=10001, user_id=1063310598, nickname="老公", timeout=90.0, settle=2.0):
        self.url = url
        self.token = token
        self.self_id = self_id
        self.user_id = user_id
        self.nickname = nickname
        self.timeout = timeout
        self.settle = settle

        self._ws = None  # 当前连接；None 表示未就绪
        self._stop = False
        self._task = None  # 常驻连接/重连循环任务

        # 单请求在飞：ui 队列已保证串行，这里只存一个 Future
        self._pending_fut = None
        self._settle_task = None  # 静默窗口定时器
        self._latest_text = None  # 静默窗口内最新一条回复

    @property
    def busy(self):
        """是否有请求在飞（等桃桃回复中）。屏幕感知门控用它避免打断对话。"""
        return self._pending_fut is not None

    # ---------- 连接管理 ----------

    def start(self):
        """启动常驻连接循环（幂等）。断线按 1→30s 指数退避自动重连。"""
        if self._task is None or self._task.done():
            self._stop = False
            self._task = current_loop().create_task(self._run())

    async def stop(self):
        """停止连接循环（幂等）：取消常驻任务并等它退出、结算在飞请求。"""
        self._stop = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # 任务被取消属预期
        self._cancel_pending()

    async def ensure_connected(self, wait=5.0):
        """等待连接就绪（含首次连接），超时抛 ConnectionError。"""
        self.start()
        deadline = current_loop().time() + wait
        while self._ws is None:
            if current_loop().time() > deadline:
                raise ConnectionError(f"无法连接 AstrBot OneBot 通道：{self.url}")
            await asyncio.sleep(0.1)

    async def _run(self):
        backoff = 1
        while not self._stop:
            try:
                # AstrBot 的 OneBot 适配器（aiocqhttp 库）握手要求三个头：
                # - Authorization: Bearer <token>（正则校验 Bearer/Token scheme）
                # - X-Client-Role: universal（连接角色：同时收事件和下发动作，缺了直接 400）
                # - X-Self-ID: <self_id>（机器人身份，服务器按它路由动作）
                # websockets>=17 用 additional_headers 传自定义头（14 曾改名 extra_headers）
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "X-Client-Role": "universal",
                    "X-Self-ID": str(self.self_id),
                }
                async with connect(self.url, additional_headers=headers) as ws:
                    self._ws = ws
                    backoff = 1
                    print(f"OneBot 通道已连接：{self.url}")
                    await ws.send(self._meta_event("lifecycle", "connect"))
                    while not self._stop:
                        try:
                            async with asyncio.timeout(30):
                                raw = await ws.recv()
                        except asyncio.TimeoutError:
                            # 30s 无消息 → 发心跳保活（OneBot 规范 meta_event）
                            await ws.send(self._meta_event("heartbeat"))
                            continue
                        self._handle(raw)
            except Exception as e:
                print(f"OneBot 通道断开：{e}，{backoff}s 后重连")
            finally:
                self._ws = None
                self._fail_pending("连接断开")
            if not self._stop:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    # ---------- 消息上报（桌宠 → AstrBot） ----------

    async def send_message(self, segments):
        """发一条 private message 事件给 AstrBot，等桃桃回复（静默窗口取最后一条）。

        返回回复纯文本；超时 / 断线 / 无回复返回空串。
        """
        await self.ensure_connected()
        raw_text = "".join(s["data"].get("text", "") for s in segments if s.get("type") == "text")
        event = {
            "time": int(time.time()),
            "self_id": self.self_id,
            "post_type": "message",
            "message_type": "private",
            "sub_type": "friend",
            "message_id": random.randint(100000, 999999),
            "user_id": self.user_id,
            "message": segments,
            "raw_message": raw_text,
            "font": 0,
            "sender": {
                "user_id": self.user_id,
                "nickname": self.nickname,
                "card": "",
                "sex": "unknown",
                "age": 0,
            },
        }
        fut = current_loop().create_future()
        if self._pending_fut is not None:  # 理论不会发生（ui 串行），防御性清理
            self._cancel_pending()
        self._pending_fut = fut
        self._latest_text = None
        try:
            await self._ws.send(json.dumps(event, ensure_ascii=False))
            return await asyncio.wait_for(fut, self.timeout)
        except asyncio.TimeoutError:
            print(f"等待桃桃回复超时（{self.timeout}s）")
            self._cancel_pending()
            return ""
        except Exception as e:
            print(f"发送消息失败：{e}")
            self._cancel_pending()
            return ""

    def _cancel_pending(self):
        # 让在飞请求尽快以空串返回（如主动观察被用户消息打断时），避免等满 timeout
        fut, self._pending_fut = self._pending_fut, None
        if fut is not None and not fut.done():
            fut.set_result("")
        if self._settle_task is not None:
            self._settle_task.cancel()
            self._settle_task = None

    def _fail_pending(self, reason):
        fut, self._pending_fut = self._pending_fut, None
        if fut is not None and not fut.done():
            fut.set_result("")

    # ---------- 动作响应（AstrBot → 桌宠） ----------

    def _handle(self, raw):
        try:
            req = json.loads(raw)
        except Exception:
            return
        action = req.get("action", "")
        params = req.get("params") or {}
        if action in ("send_private_msg", "send_group_msg", "send_msg"):
            # 桃桃的回复：文本进结算，图片等非文本部分丢弃
            text = extract_message_text(params.get("message"))
            if text:
                self._on_reply(text)
            self._reply(req, {"message_id": random.randint(100000, 999999)})
        elif action == "get_login_info":
            # AstrBot 靠这个识别 bot 身份，必须返回稳定 id
            self._reply(req, {"user_id": self.self_id, "nickname": "糯糯"})
        else:
            # 其余动作（get_friend_list 等）回 ok + 空数据，避免适配器报错
            self._reply(req, {})

    def _reply(self, req, data):
        if self._ws is None:
            return
        resp = {"status": "ok", "retcode": 0, "data": data}
        if req.get("echo") is not None:
            resp["echo"] = req["echo"]
        current_loop().create_task(self._ws.send(json.dumps(resp, ensure_ascii=False)))

    # ---------- 回复结算（静默窗口） ----------

    def _on_reply(self, text):
        """收到一条回复：记入最新文本，重启静默窗口；settle 秒无新回复则结算。"""
        self._latest_text = text
        if self._settle_task is not None:
            self._settle_task.cancel()
        self._settle_task = current_loop().create_task(self._settle_after(self.settle))

    async def _settle_after(self, delay):
        await asyncio.sleep(delay)
        fut, self._pending_fut = self._pending_fut, None
        if fut is not None and not fut.done():
            fut.set_result(self._latest_text or "")

    # ---------- 心跳 / 生命周期事件 ----------

    def _meta_event(self, meta_type, sub_type=None):
        event = {
            "time": int(time.time()),
            "self_id": self.self_id,
            "post_type": "meta_event",
            "meta_event_type": meta_type,
        }
        if sub_type:
            event["sub_type"] = sub_type
        if meta_type == "heartbeat":
            event["status"] = {"online": True, "good": True}
            event["interval"] = 30000
        return json.dumps(event, ensure_ascii=False)
