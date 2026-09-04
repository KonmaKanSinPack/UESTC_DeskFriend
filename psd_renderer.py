"""桃桃的分层渲染器（PSD 直驱，2026-09-04）：吃 manifest + 层 PNG，画"会动的桃桃"。

架构边界：渲染方式是皮（skin）的内部细节——本模块不 import 任何器官/大脑，
只暴露 `PsdRenderer(QWidget)`：`set_state(state)` 语义级命令 + 自驱动画定时器。
鼠标位置是纯输入读取（同耳读麦克风），不是对其它器官的依赖。

动画清单（对应 desktop-pet 领域的"活感"来源，按贡献排序）：
- 眨眼：左右眼独立、随机 2~6s 间隔、~130ms 一眨（生命感最大单项）
- 视线跟随：瞳孔（irides）在眼窝内追全局鼠标位置（钳制幅度）
- 口型：talking 态嘴巴 scaleY 正弦开合（不需要画多套口型的便宜方案）
- 呼吸：躯干以上微幅起伏；发丝轻摆：前后发反相位微旋转
- thinking：头部组小幅摆动（比整窗晃动精细一个层级）

窗口不动、只在控件内重绘——天然避开"每帧 move 整窗触发 DWM 全窗重合成"
的最贵动画路径（2026-09-03 调研结论在渲染器路线上顺带落实）。
"""

import json
import math
import random
import time
from pathlib import Path

from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import QCursor, QPainter, QPixmap, QTransform
from PyQt5.QtWidgets import QWidget

# 显示宽度：768 画布缩到 220——眼睛区域约 20px，视线偏移 ±2px 可感知
DISPLAY_WIDTH = 220
FPS = 30  # 渲染帧率：眨眼/口型需要；idle 态开销也仅为控件内重绘（窗口不动）

HEAD_PARTS = {"face", "mouth", "nose", "ears", "headwear", "front hair", "back hair"}
EYE_PART_KEYWORDS = ("eyelash", "eyewhite", "irides", "eyebrow")
IRIS_NAMES = ("irides l", "irides r")
BLINK_MIN_S, BLINK_MAX_S = 2.0, 6.0
BLINK_DURATION_S = 0.13
EYE_TRACK_PX = 14.0  # 瞳孔最大偏移（768 画布坐标系）
MOUTH_OSC_HZ = 6.0  # 说话口型开合频率


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def blink_scale(t_since_start: float, duration: float = BLINK_DURATION_S) -> float:
    """眨眼曲线：1 → 0.05 → 1 的下凹抛物线（0.05 而非 0：留一条缝更像眯眼）。"""
    if t_since_start < 0 or t_since_start > duration:
        return 1.0
    # 半程对称的 (1 - sin(π·t/T)) 变体，谷底 0.05
    return 1.0 - 0.95 * math.sin(math.pi * t_since_start / duration)


def eye_track_offset(target_dx: float, target_dy: float, center_dist: float, range_px: float = EYE_TRACK_PX):
    """视线偏移向量：朝鼠标方向取 range_px，越近目标幅度越小（贴近时注视）。"""
    scale = clamp(center_dist / 400.0, 0.0, 1.0)  # 400px 外达到满幅；贴脸时收敛
    mag = math.hypot(target_dx, target_dy)
    if mag < 1e-6:
        return 0.0, 0.0
    return (target_dx / mag) * range_px * scale, (target_dy / mag) * range_px * scale


def mouth_scale(t: float, talking: bool) -> float:
    """口型：talking 时 0.55~1.45 正弦开合，其余闭合（1.0）。"""
    if not talking:
        return 1.0
    return 1.0 + 0.45 * math.sin(2 * math.pi * MOUTH_OSC_HZ * t)


class _Layer:
    """一层：贴图 + 摆放位置 + 分组归类（按层名推断，manifest 无需手标）。"""

    def __init__(self, name, pixmap, x, y):
        self.name = name
        self.pixmap = pixmap
        self.x = x
        self.y = y
        low = name.lower()
        self.side = "L" if low.endswith(" l") else ("R" if low.endswith(" r") else None)
        base = low.rstrip(" lr").strip()
        self.is_eye = any(k in base for k in EYE_PART_KEYWORDS)
        self.is_iris = low in IRIS_NAMES
        self.is_mouth = base == "mouth"
        self.is_front_hair = base == "front hair"
        self.is_back_hair = base == "back hair"
        self.is_head = base in HEAD_PARTS or self.is_eye


class PsdRenderer(QWidget):
    """分层渲染的桌宠本体：load(manifest) → set_state(idle/thinking/talking) 即活。"""

    def __init__(self, manifest_path, parent=None):
        super().__init__(parent)
        self.manifest_path = Path(manifest_path)
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.canvas_w, self.canvas_h = data["canvas"]
        self.layers = [
            _Layer(entry["name"], QPixmap(str(self.manifest_path.parent / entry["file"])), entry["x"], entry["y"])
            for entry in data["layers"]
        ]
        # 显示尺寸：按 DISPLAY_WIDTH 等比缩放整只
        self.scale = DISPLAY_WIDTH / self.canvas_w
        self.setFixedSize(int(self.canvas_w * self.scale), int(self.canvas_h * self.scale))
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.state = "idle"
        self._t0 = time.monotonic()
        # 每只眼独立的眨眼调度（L/R 各一列：下次眨眼时刻 + 眨眼起点）
        self._next_blink = {"L": self._rand_blink_at(), "R": self._rand_blink_at()}
        self._blink_start = {"L": None, "R": None}
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000 / FPS))

    # ---------- 对外 ----------

    def set_state(self, state):
        """语义级命令（与旧整窗动画同一语义）：idle/thinking/talking。"""
        self.state = state

    # ---------- 内部 ----------

    def _rand_blink_at(self):
        return time.monotonic() + random.uniform(BLINK_MIN_S, BLINK_MAX_S)

    def _tick(self):
        self.update()  # 只重绘控件（窗口不动），触发 paintEvent

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        now = time.monotonic()
        t = now - self._t0

        # 视线目标：全局鼠标 → 本控件中心的方向（纯输入读取）。
        # isVisible 守卫：未 show 的控件没有窗口句柄，mapToGlobal 是原生崩溃
        # （真机常显示，此守卫为离屏渲染/paint 先于 show 的防御）
        track = (0.0, 0.0)
        if self.isVisible():
            center = self.mapToGlobal(QPointF(self.width() / 2, self.height() / 2))
            cursor = QCursor.pos()
            dx, dy = cursor.x() - center.x(), cursor.y() - center.y()
            track = eye_track_offset(dx, dy, math.hypot(dx, dy))
            # 鼠标在本控件内时不追踪（贴脸注视会斗鸡眼，收敛为直视）
            if self.rect().contains(self.mapFromGlobal(cursor)):
                track = (0.0, 0.0)

        # thinking：头部组绕颈点小幅摆动（度）+ 上浮（画布坐标）——所有头部件
        # 共用同一锚点旋转，层与层才不会剪切错位
        head_rot = math.sin(t * 6.0) * 4.0 if self.state == "thinking" else 0.0
        head_bob = -abs(math.sin(t * 3.0)) * 6.0 if self.state == "thinking" else 0.0
        breathe = math.sin(t * 1.6) * 3.0  # 呼吸：躯干以上缓起伏（画布坐标 y）
        head_pivot = (self.canvas_w / 2, self.canvas_h * 0.58)  # 颈点≈头组旋转锚

        for layer in self.layers:
            x, y, rot, sx, sy = float(layer.x), float(layer.y), 0.0, 1.0, 1.0
            if layer.is_iris and track != (0.0, 0.0):
                x += track[0]
                y += track[1]
            if layer.is_eye and layer.side:
                # 眨眼：该侧到期 → 触发；进行中 → 眼组 scaleY（锚定眼顶部下压）
                start = self._blink_start[layer.side]
                if start is None and now >= self._next_blink[layer.side]:
                    start = self._blink_start[layer.side] = now
                if start is not None:
                    sy = blink_scale(now - start)
                    if now - start > BLINK_DURATION_S:
                        self._blink_start[layer.side] = None
                        self._next_blink[layer.side] = self._rand_blink_at()
                if sy < 1.0:
                    # 锚顶：把 y 下移 (1-sy)×层高，缩放后顶边不动
                    y += (1.0 - sy) * layer.pixmap.height()
            if layer.is_mouth:
                sy = mouth_scale(t, self.state == "talking")
                if sy != 1.0:
                    y -= (sy - 1.0) * layer.pixmap.height() * 0.5  # 锚中开合
            if layer.is_head:
                y += head_bob + breathe
                rot = head_rot
            if layer.is_front_hair:
                rot += math.sin(t * 0.9) * 2.0  # 发丝摆动叠在头部姿态上（同锚点）
            elif layer.is_back_hair:
                rot += -math.sin(t * 0.9) * 1.2  # 反相位更"软"

            tr = QTransform()
            tr.scale(self.scale, self.scale)
            if rot:
                # 会旋转的只有头部件/头发：统一绕颈点摆——逐层绕自身中心会剪切散架
                cx, cy = head_pivot
                tr.translate(cx, cy)
                tr.rotate(rot)
                tr.translate(-cx, -cy)
            tr.translate(x, y)
            tr.scale(sx, sy)
            painter.setTransform(tr)
            painter.drawPixmap(0, 0, layer.pixmap)
