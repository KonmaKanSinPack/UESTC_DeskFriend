"""OneBotBridge 协议测试：进程内起一个假 WS 服务端，扮演 AstrBot 的 OneBot 适配器。

覆盖：连接握手与鉴权头、message 事件格式、动作响应（send_private_msg / get_login_info /
未知动作 / echo 对账）、静默窗口结算、超时兜底。

注意：handler 必须跳过 lifecycle/meta 事件并保持连接存活（提前返回会让桥认为断线重连）。
"""

import asyncio
import json

from websockets.asyncio.server import serve

from onebot_bridge import OneBotBridge, extract_forward_nodes_text, extract_message_text

TEST_TOKEN = "test-token"


async def _start_server(handler):
    server = await serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


def _make_bridge(port, settle=0.1, timeout=2.0):
    return OneBotBridge(
        f"ws://127.0.0.1:{port}",
        token=TEST_TOKEN,
        self_id=10001,
        user_id=1063310598,
        nickname="老公",
        timeout=timeout,
        settle=settle,
    )


def _action(action, params=None, echo="e1"):
    return json.dumps({"action": action, "params": params or {}, "echo": echo})


async def _wait_message_event(ws):
    """跳过 lifecycle/meta 事件，等到第一条 message 事件并返回。"""
    while True:
        evt = json.loads(await ws.recv())
        if evt.get("post_type") == "message":
            return evt


def test_connect_sends_lifecycle_and_token():
    """连接后：握手带 Bearer token，随后上报 lifecycle/connect 事件。"""

    async def run():
        done = asyncio.Event()
        seen = {"headers": None, "frames": []}

        async def handler(ws):
            seen["headers"] = dict(ws.request.headers)
            seen["frames"].append(json.loads(await ws.recv()))
            await done.wait()  # 保持连接存活，避免测试中途断线干扰桥的重连逻辑

        server, port = await _start_server(handler)
        bridge = _make_bridge(port)
        bridge.start()
        for _ in range(100):
            if seen["frames"]:
                break
            await asyncio.sleep(0.05)
        await bridge.stop()
        done.set()
        server.close()
        await server.wait_closed()
        await asyncio.sleep(0.2)  # 让桥的连接任务干净退出

        # websockets 会把请求头名统一转为小写（HTTP 头名大小写不敏感）
        assert seen["headers"].get("authorization") == f"Bearer {TEST_TOKEN}"
        evt = seen["frames"][0]
        assert evt["post_type"] == "meta_event"
        assert evt["meta_event_type"] == "lifecycle"
        assert evt["sub_type"] == "connect"
        assert evt["self_id"] == 10001

    asyncio.run(run())


def test_send_message_event_format_and_reply():
    """send_message：message 事件格式正确，回复经 send_private_msg 动作提取文本返回。"""

    async def run():
        received = []

        async def handler(ws):
            evt = await _wait_message_event(ws)
            received.append(evt)
            # 扮演 AstrBot：下发 send_private_msg 动作，等桥的 ok 响应
            await ws.send(_action("send_private_msg", {"user_id": 1063310598, "message": "你好呀"}))
            await ws.recv()
            await asyncio.sleep(0.3)  # 等静默窗口结算完再断开

        server, port = await _start_server(handler)
        bridge = _make_bridge(port)
        reply = await bridge.send_message([{"type": "text", "data": {"text": "糯糯在吗"}}])
        await bridge.stop()
        server.close()
        await server.wait_closed()

        evt = received[0]
        assert evt["post_type"] == "message"
        assert evt["message_type"] == "private"
        assert evt["user_id"] == 1063310598
        assert evt["self_id"] == 10001
        assert evt["sender"]["nickname"] == "老公"
        assert evt["message"] == [{"type": "text", "data": {"text": "糯糯在吗"}}]
        assert evt["raw_message"] == "糯糯在吗"
        assert reply == "你好呀"

    asyncio.run(run())


def test_action_responses():
    """动作响应：get_login_info 返回伪装身份；未知动作回 ok+空；echo 原样回传。"""

    async def run():
        responses = []

        async def handler(ws):
            await _wait_message_event(ws)  # 等一条消息事件才开始动作对答
            await ws.send(_action("get_login_info", echo="e1"))
            responses.append(json.loads(await ws.recv()))
            await ws.send(_action("get_friend_list", echo="e2"))
            responses.append(json.loads(await ws.recv()))

        server, port = await _start_server(handler)
        bridge = _make_bridge(port)
        task = asyncio.create_task(bridge.send_message([{"type": "text", "data": {"text": "hi"}}]))
        for _ in range(100):
            if len(responses) == 2:
                break
            await asyncio.sleep(0.05)
        await bridge.stop()
        server.close()
        await server.wait_closed()
        await task

        login, friends = responses
        assert login["status"] == "ok"
        assert login["data"]["user_id"] == 10001
        assert login["echo"] == "e1"
        assert friends["status"] == "ok"
        assert friends["echo"] == "e2"

    asyncio.run(run())


def test_settle_window_takes_last_reply():
    """静默窗口：多条回复间隔在窗口内（占位 + 最终）→ 窗口每次重置，取最后一条。"""

    async def run():
        async def handler(ws):
            await _wait_message_event(ws)
            await ws.send(_action("send_private_msg", {"message": "已收到，正在思考中…"}, echo="e1"))
            await asyncio.sleep(0.15)  # 小于 settle(0.3)：重置窗口
            await ws.send(_action("send_private_msg", {"message": "我觉得是这样"}, echo="e2"))
            await asyncio.sleep(0.4)  # 等结算完再断开

        server, port = await _start_server(handler)
        bridge = _make_bridge(port, settle=0.3)
        reply = await bridge.send_message([{"type": "text", "data": {"text": "你怎么看"}}])
        await bridge.stop()
        server.close()
        await server.wait_closed()

        assert reply == "我觉得是这样"

    asyncio.run(run())


def test_settle_window_earlier_reply_when_gap_exceeds():
    """已知边界：回复间隔超过 settle → 窗口已结算，返回较早的回复。

    这正是 AstrBot 开「回复中提示」时要把 ASTRBOT_SETTLE 调大的原因——
    占位与最终回复的间隔超过静默窗口，占位就会成为"最终"回复。
    """

    async def run():
        async def handler(ws):
            await _wait_message_event(ws)
            await ws.send(_action("send_private_msg", {"message": "已收到，正在思考中…"}, echo="e1"))
            await asyncio.sleep(0.3)  # 超过 settle(0.1)：窗口已结算
            await ws.send(_action("send_private_msg", {"message": "真正的回复"}, echo="e2"))
            await asyncio.sleep(0.2)

        server, port = await _start_server(handler)
        bridge = _make_bridge(port, settle=0.1)
        reply = await bridge.send_message([{"type": "text", "data": {"text": "你怎么看"}}])
        await bridge.stop()
        server.close()
        await server.wait_closed()

        assert reply == "已收到，正在思考中…"

    asyncio.run(run())


def test_timeout_returns_empty():
    """服务端不回复：send_message 在 timeout 后返回空串，不挂起。"""

    async def run():
        async def handler(ws):
            await _wait_message_event(ws)
            await asyncio.sleep(1.0)  # 收下事件但不回任何动作，保持连接

        server, port = await _start_server(handler)
        bridge = _make_bridge(port, timeout=0.4)
        reply = await bridge.send_message([{"type": "text", "data": {"text": "在吗"}}])
        await bridge.stop()
        server.close()
        await server.wait_closed()

        assert reply == ""

    asyncio.run(run())


def test_image_segment_in_event():
    """图片消息段：base64 图片原样进 message 事件。"""

    async def run():
        received = []

        async def handler(ws):
            evt = await _wait_message_event(ws)
            received.append(evt)
            await ws.send(_action("send_private_msg", {"message": "看到了"}))
            await ws.recv()
            await asyncio.sleep(0.3)

        server, port = await _start_server(handler)
        bridge = _make_bridge(port)
        segments = [
            {"type": "text", "data": {"text": "看图"}},
            {"type": "image", "data": {"file": "base64://AAAA", "subType": 0}},
        ]
        reply = await bridge.send_message(segments)
        await bridge.stop()
        server.close()
        await server.wait_closed()

        assert received[0]["message"] == segments
        assert received[0]["raw_message"] == "看图"  # 图片不算进 raw_message
        assert reply == "看到了"

    asyncio.run(run())


def test_forward_msg_action_extracts_nodes_and_replies():
    """合并转发动作（Splitter 类插件用它打包多条消息）：node 文本拼接进结算，回规范 message_id。

    曾踩坑：把 send_private_forward_msg 当未知动作回空 ok → AstrBot 侧拿不到有效
    data → Splitter 报"发送失败"。
    """

    async def run():
        async def handler(ws):
            evt = await _wait_message_event(ws)
            assert evt["message"][0]["data"]["text"] == "测试长回复"
            nodes = [
                {
                    "type": "node",
                    "data": {"uin": 10001, "name": "桃桃", "content": [{"type": "text", "data": {"text": "第一段"}}]},
                },
                {
                    "type": "node",
                    "data": {"uin": 10001, "name": "桃桃", "content": [{"type": "text", "data": {"text": "第二段"}}]},
                },
            ]
            await ws.send(_action("send_private_forward_msg", {"user_id": 1063310598, "messages": nodes}, echo="e1"))
            resp = json.loads(await ws.recv())
            assert resp["status"] == "ok"
            assert resp["echo"] == "e1"
            assert isinstance(resp["data"]["message_id"], int)
            await asyncio.sleep(0.3)  # 保持连接，等静默窗口结算完再断开

        server, port = await _start_server(handler)
        bridge = _make_bridge(port)
        reply = await bridge.send_message([{"type": "text", "data": {"text": "测试长回复"}}])
        await bridge.stop()
        server.close()
        await server.wait_closed()

        assert reply == "第一段\n第二段"

    asyncio.run(run())


class TestExtractMessageText:
    def test_str_passthrough(self):
        assert extract_message_text("直接文本") == "直接文本"

    def test_segments_array(self):
        msg = [
            {"type": "text", "data": {"text": "你好"}},
            {"type": "image", "data": {"file": "x"}},
            {"type": "text", "data": {"text": "呀"}},
        ]
        assert extract_message_text(msg) == "你好呀"

    def test_non_text(self):
        assert extract_message_text([{"type": "image", "data": {"file": "x"}}]) == ""
        assert extract_message_text(None) == ""


class TestExtractForwardNodes:
    def test_nodes_joined_by_newline(self):
        nodes = [
            {"type": "node", "data": {"content": [{"type": "text", "data": {"text": "甲"}}]}},
            {
                "type": "node",
                "data": {
                    "content": [{"type": "text", "data": {"text": "乙"}}, {"type": "text", "data": {"text": "丙"}}]
                },
            },
        ]
        assert extract_forward_nodes_text(nodes) == "甲\n乙丙"

    def test_empty_and_non_list(self):
        assert extract_forward_nodes_text([]) == ""
        assert extract_forward_nodes_text(None) == ""
        assert extract_forward_nodes_text([{"type": "text", "data": {"text": "不是节点"}}]) == ""

    def test_node_without_text_skipped(self):
        nodes = [
            {"type": "node", "data": {"content": [{"type": "image", "data": {"file": "x"}}]}},
            {"type": "node", "data": {"content": [{"type": "text", "data": {"text": "只有这句"}}]}},
        ]
        assert extract_forward_nodes_text(nodes) == "只有这句"
