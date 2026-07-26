"""Wayland 静默截屏后端：XDG ScreenCast Portal + PipeWire + GStreamer。

gnome-screenshot 每次截屏都有闪光灯效，Wayland 下 Pillow 的 ImageGrab
又不可用，所以走 ScreenCast 门户申请一条持久的 PipeWire 屏幕流，按需拉帧。
首次运行会弹一次 GNOME 屏幕共享授权框；用户允许后把 restore_token 存到
项目根目录的 pw_restore_token，之后启动完全静默。
"""

import os
import threading
import uuid

import gi
from PIL import Image

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import (  # noqa: E402
    Gio,
    GLib,
    Gst,
    GstApp,  # noqa: E402,F401  # 导入后 GstAppSink 才有 try_pull_sample 等方法
    GstVideo,
)

_PORTAL_BUS = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_SC_IFACE = "org.freedesktop.portal.ScreenCast"
_REQ_IFACE = "org.freedesktop.portal.Request"

_TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pw_restore_token")


class PipeWireCapture:
    """按需拉帧的静默截屏后端；所有环节失败都安静回退，不向外抛异常。"""

    def __init__(self):
        self._init_done = threading.Event()  # 初始化流程结束（无论成败）
        self._ok = False  # 初始化成功、随时可抓帧
        self._started = False  # 后台初始化是否已启动
        self._loop = None  # 跑在守护线程里的 GLib 主循环（Portal 的 Response 信号靠它分发）
        self._conn = None
        self._session_handle = None
        self._pipeline = None
        self._appsink = None
        self._handlers = {}  # request token -> Response 信号的处理回调
        self._watchdog = 0  # 授权超时定时器的 source id
        self._grab_lock = threading.Lock()  # 抓帧时暂停/恢复管线要串行

    # ---------- 对外接口 ----------

    def start(self):
        """惰性启动后台初始化（首次截屏时才走授权流程），可重复调用。"""
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._init_worker, name="pw-capture-init", daemon=True).start()

    @property
    def pending(self):
        """初始化流程未结束（含尚未开始、等待用户授权）。"""
        return not self._init_done.is_set()

    @property
    def ok(self):
        """初始化成功、随时可抓帧。"""
        return self._ok

    def grab(self, init_wait=3.0):
        """抓一帧屏幕内容，返回 PIL.Image；后端未就绪或抓帧失败返回 None。

        init_wait：最多等初始化完成的秒数。正常冷启动（token 已存在）只需
        零点几秒；若是在等用户点授权框则会超时返回 None，由上层跳过本次。
        不能让 UI 线程无限等授权，所以必须有界。
        """
        try:
            self.start()
            if not self._init_done.wait(timeout=init_wait) or not self._ok:
                return None
            with self._grab_lock:
                # 管线平时停在 PAUSED 省 CPU，抓帧时才恢复；GNOME 在流恢复时会立刻补一帧
                self._pipeline.set_state(Gst.State.PLAYING)
                self._appsink.try_pull_sample(0)  # 丢掉暂停前可能残留的过期帧
                sample = self._appsink.try_pull_sample(2 * Gst.SECOND)
                self._pipeline.set_state(Gst.State.PAUSED)
            return self._sample_to_image(sample) if sample else None
        except Exception:
            return None

    # ---------- 初始化（Portal 授权 + 建管线，全在守护线程里跑） ----------

    def _init_worker(self):
        try:
            Gst.init(None)
            self._loop = GLib.MainLoop()
            GLib.idle_add(self._begin_portal_flow)
            self._loop.run()
        except Exception as e:
            print(f"PipeWire 截屏初始化失败：{e}")
            self._fail()

    def _begin_portal_flow(self):
        try:
            self._conn = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            # 订阅所有 Request 对象的 Response 信号，按 token 分发回我们自己的请求
            self._conn.signal_subscribe(
                _PORTAL_BUS,
                _REQ_IFACE,
                "Response",
                None,
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_response,
            )
            # 用户迟迟不点授权框时放弃，让上层回退 gnome-screenshot
            self._watchdog = GLib.timeout_add_seconds(120, self._on_init_timeout)
            token = self._new_token()
            self._handlers[token] = self._on_session_response
            options = {
                "handle_token": GLib.Variant("s", token),
                "session_handle_token": GLib.Variant("s", self._new_token()),
            }
            self._call(_SC_IFACE, "CreateSession", GLib.Variant("(a{sv})", (options,)), "(o)", self._on_empty_reply)
        except Exception as e:
            print(f"PipeWire 截屏初始化失败：{e}")
            self._fail()
        return False  # idle_add 只执行一次

    def _call(self, iface, method, params, reply_type, callback):
        self._conn.call(
            _PORTAL_BUS,
            _PORTAL_PATH,
            iface,
            method,
            params,
            GLib.VariantType(reply_type) if reply_type else None,
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            callback,
        )

    def _on_session_response(self, response, results):
        # CreateSession 的应答只是个 Request 句柄，session_handle 要等它的 Response 信号
        if response != 0 or not results.get("session_handle"):
            self._fail()
            return
        self._session_handle = results["session_handle"]
        token = self._new_token()
        self._handlers[token] = self._on_sources_selected
        options = {
            "handle_token": GLib.Variant("s", token),  # 不带的话 Response 路径上是 portal 自造的 token，没法分发
            "types": GLib.Variant("u", 1),  # 只共享显示器
            "multiple": GLib.Variant("b", False),  # 只要一路流
            "persist_mode": GLib.Variant("u", 2),  # 记住授权，以后不再弹窗
        }
        saved = self._load_token()
        if saved:
            options["restore_token"] = GLib.Variant("s", saved)
        params = GLib.Variant("(oa{sv})", (self._session_handle, options))
        self._call(_SC_IFACE, "SelectSources", params, None, self._on_empty_reply)

    def _on_sources_selected(self, response, results):
        if response != 0:  # 用户点了取消
            self._fail()
            return
        token = self._new_token()
        self._handlers[token] = self._on_started
        options = {"handle_token": GLib.Variant("s", token)}
        params = GLib.Variant("(osa{sv})", (self._session_handle, "", options))
        self._call(_SC_IFACE, "Start", params, None, self._on_empty_reply)

    def _on_started(self, response, results):
        try:
            if response != 0 or not results.get("streams"):
                self._fail()
                return
            node_id = results["streams"][0][0]  # streams: a(ua{sv})，取第一路流的 PipeWire node id
            if results.get("restore_token"):
                self._save_token(results["restore_token"])
            fd = self._open_pipewire_remote()
            self._build_pipeline(fd, node_id)
            self._ok = True
            print("PipeWire 静默截屏已启用")
            self._finish()
        except Exception as e:
            print(f"PipeWire 截屏初始化失败：{e}")
            self._fail()

    def _open_pipewire_remote(self):
        params = GLib.Variant("(oa{sv})", (self._session_handle, {}))
        result, fd_list = self._conn.call_with_unix_fd_list_sync(
            _PORTAL_BUS,
            _PORTAL_PATH,
            _SC_IFACE,
            "OpenPipeWireRemote",
            params,
            GLib.VariantType("(h)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            None,
        )
        return fd_list.get(result.unpack()[0])  # 按返回的句柄序号取出真正的 fd

    def _build_pipeline(self, fd, node_id):
        # 屏幕原始格式（一般是 BGRx）统一转成 RGB，appsink 只留最新一帧
        desc = (
            f"pipewiresrc fd={fd} path={node_id} ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=shot_sink max-buffers=1 drop=true sync=false"
        )
        self._pipeline = Gst.parse_launch(desc)
        self._appsink = self._pipeline.get_by_name("shot_sink")
        self._pipeline.set_state(Gst.State.PLAYING)
        # 等首帧确认流真的通了，然后暂停待命（抓帧时再恢复）
        self._appsink.try_pull_sample(5 * Gst.SECOND)
        self._pipeline.set_state(Gst.State.PAUSED)

    # ---------- 信号与回调 ----------

    def _on_response(self, conn, sender, path, iface, signal, params):
        response, results = params.unpack()
        token = path.rsplit("/", 1)[-1]  # 请求路径末段就是我们发的 handle_token
        handler = self._handlers.pop(token, None)
        if handler is not None:
            handler(response, results)

    def _on_empty_reply(self, conn, result):
        try:
            conn.call_finish(result)
        except Exception as e:
            print(f"PipeWire 截屏初始化失败：{e}")
            self._fail()

    def _on_init_timeout(self):
        self._watchdog = 0
        print("PipeWire 截屏授权超时，回退到 gnome-screenshot")
        self._fail()
        return False  # 定时器只触发一次

    def _finish(self):
        if self._watchdog:
            GLib.source_remove(self._watchdog)
            self._watchdog = 0
        self._init_done.set()
        if self._loop is not None:
            self._loop.quit()  # 授权流程结束，D-Bus 主循环可以退场了

    def _fail(self):
        self._ok = False
        self._finish()

    # ---------- 帧与 token 的小工具 ----------

    @staticmethod
    def _sample_to_image(sample):
        info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        width, height, stride = info.width, info.height, info.stride[0]
        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            data = map_info.data
            if stride == width * 3:
                return Image.frombytes("RGB", (width, height), data)
            # 有行对齐填充时逐行裁掉
            rows = (data[y * stride : y * stride + width * 3] for y in range(height))
            return Image.frombytes("RGB", (width, height), b"".join(rows))
        finally:
            buf.unmap(map_info)

    @staticmethod
    def _load_token():
        try:
            with open(_TOKEN_FILE) as f:
                return f.read().strip() or None
        except OSError:
            return None

    @staticmethod
    def _save_token(token):
        try:
            with open(_TOKEN_FILE, "w") as f:
                f.write(token + "\n")
        except OSError:
            pass

    @staticmethod
    def _new_token():
        return "deskfriend_" + uuid.uuid4().hex[:8]
