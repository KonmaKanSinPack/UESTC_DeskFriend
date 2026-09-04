"""一次性 PSD 预处理工具：See-through 拆层产物 → 渲染器可直接吃的层 PNG + manifest。

三步（2026-09-04 PSD 直驱轮，背景见 docs_agent/session/2026-09-04.md 第三轮）：
1. 左右分离：双眼合一的层（eyewhite/irides/eyelash/eyebrow/handwear/footwear/legwear）
   按"列间隙"切成 L/R 两层——生命感的核心就是左右眼独立动
2. 边缘扩展：每层向外扩 ~15px、新像素取四邻实色（迭代传播）——部件位移不露洞
   （拆层只是切开，被遮挡的背后是空的）
3. 导出：输出目录一层一 PNG + manifest.json（canvas/层列表/底层→顶层绘制序）

用法：uv run python tools/prepare_psd.py <输入.psd> <输出目录>
依赖 psd-tools（仅本工具用；运行时渲染器只吃 PNG+manifest）。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from psd_tools import PSDImage

# 这些层是"左右对称部件合一层"，需要拆分（See-through 拆层器的已知局限）
SPLIT_LAYER_NAMES = {"eyewhite", "irides", "eyelash", "eyebrow", "handwear", "footwear", "legwear"}
EDGE_EXTEND_PX = 15  # 渲染位移幅度一般 <10px，15 留余量


def split_lr(arr: np.ndarray):
    """把左右两块内容合一的 RGBA 层按"最大列间隙"切成 (左, 右, 右块列起点)。

    原理/复现：左右部件（两眼/两手）之间必有一段全透明列；投影每列 alpha 和，
    在内容首尾列之间找最长连续零段、取其中点下刀。找不到间隙（单块内容）
    返回 None——说明该层无需拆。
    """
    alpha_col = arr[:, :, 3].sum(axis=0) > 0
    cols = np.where(alpha_col)[0]
    if len(cols) == 0:
        return None
    lo, hi = cols[0], cols[-1]
    best_len, best_start, run, run_start = 0, None, 0, None
    for x in range(lo, hi + 1):
        if not alpha_col[x]:
            if run == 0:
                run_start = x
            run += 1
            if run > best_len:
                best_len, best_start = run, run_start
        else:
            run = 0
    if best_len == 0:
        return None
    cut = best_start + best_len // 2
    return arr[:, :cut, :], arr[:, cut:, :], cut


def edge_extend(arr: np.ndarray, px: int = EDGE_EXTEND_PX) -> np.ndarray:
    """把 RGBA 层向外扩 px 像素，新像素取四邻实色（roll 迭代传播）。

    原理：每轮把每个实像素（含颜色）向四个方向推一格、只填空位——
    px 轮 ≈ 曼哈顿距离 px 的边缘内容自然外推，无需图像修复模型。
    先四周补 px 空边，roll 的回绕只会发生在全空的补边区（内容距边 ≥px），
    不会污染结果。
    """
    h, w = arr.shape[:2]
    out = np.zeros((h + 2 * px, w + 2 * px, 4), dtype=np.float32)
    out[px : px + h, px : px + w] = arr.astype(np.float32)
    solid = out[:, :, 3] > 0
    for _ in range(px):
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            rolled = np.roll(out, shift, axis=axis)
            rolled_solid = np.roll(solid, shift, axis=axis)
            fill = rolled_solid & ~solid
            if fill.any():
                out[fill] = rolled[fill]
                solid |= fill
    return out.astype(np.uint8)


def prepare(psd_path: str, out_dir: str):
    psd = PSDImage.open(psd_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"canvas": [psd.width, psd.height], "layers": []}
    for layer in psd:  # psd 迭代序 = 底层→顶层 = 绘制序
        if layer.is_group():
            continue
        name = layer.name.strip()
        pil = layer.composite().convert("RGBA")
        arr = np.array(pil)
        x, y = layer.offset
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            continue

        pieces = [(name, arr, x, y)]
        if name.lower() in SPLIT_LAYER_NAMES:
            lr = split_lr(arr)
            if lr is not None:
                left, right, cut = lr
                pieces = [
                    (f"{name} L", left, x, y),
                    (f"{name} R", right, x + cut, y),
                ]

        for pname, piece, px_, py_ in pieces:
            extended = edge_extend(piece)
            eh, ew = extended.shape[:2]
            if eh <= 2 * EDGE_EXTEND_PX or ew <= 2 * EDGE_EXTEND_PX:
                continue  # 拆分/扩展后仍全空
            fname = pname.replace(" ", "_") + ".png"
            Image.fromarray(extended).save(out / fname)
            manifest["layers"].append(
                # x/y 记扩边后的左上角（原点外移 px），渲染时直接按此摆放
                {"name": pname, "file": fname, "x": px_ - EDGE_EXTEND_PX, "y": py_ - EDGE_EXTEND_PX}
            )
            print(f"  ✓ {pname}  {ew}x{eh}")

    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {len(manifest['layers'])} 层 → {out / 'manifest.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    prepare(sys.argv[1], sys.argv[2])
