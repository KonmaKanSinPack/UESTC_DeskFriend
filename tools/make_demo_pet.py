"""演示宠物合成器：不依赖真 PSD，生成一套色块宠物层 + manifest，端到端验证 PsdRenderer。

真 PSD 回归前可用来先行验证渲染链路（眨眼/视线/口型/呼吸）；真素材到位后用
tools/prepare_psd.py 重新生成 assets/live2d/taotao/ 即可，两工具输出同 schema。

用法：uv run python tools/make_demo_pet.py <输出目录>
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

CANVAS = 768, 768


def _ellipse_layer(w, h, color, out_path):
    """椭圆色块层（透明底）。"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([0, 0, w - 1, h - 1], fill=color)
    img.save(out_path)
    return img


def make_demo(out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, w, h, color, x, y):
        _ellipse_layer(w, h, color, out / f"{name.replace(' ', '_')}.png")
        return {"name": name, "file": f"{name.replace(' ', '_')}.png", "x": x, "y": y}

    layers = [
        save("back hair", 420, 380, (90, 60, 70, 255), 174, 120),
        save("face", 300, 280, (255, 228, 214, 255), 234, 190),
        save("front hair", 340, 150, (70, 45, 55, 255), 214, 110),
        save("eyewhite L", 60, 34, (255, 255, 255, 255), 268, 300),
        save("eyewhite R", 60, 34, (255, 255, 255, 255), 440, 300),
        save("irides L", 34, 30, (120, 60, 90, 255), 281, 302),
        save("irides R", 34, 30, (120, 60, 90, 255), 453, 302),
        save("eyelash L", 64, 16, (50, 30, 40, 255), 266, 292),
        save("eyelash R", 64, 16, (50, 30, 40, 255), 438, 292),
        save("eyebrow L", 56, 10, (60, 35, 45, 255), 270, 274),
        save("eyebrow R", 56, 10, (60, 35, 45, 255), 442, 274),
        save("mouth", 26, 14, (200, 90, 110, 255), 371, 396),
        save("topwear", 260, 200, (240, 130, 140, 255), 254, 440),
    ]
    manifest = {"canvas": list(CANVAS), "layers": layers}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"演示宠物：{len(layers)} 层 → {out / 'manifest.json'}")
    print(f"验证：把 config/common.toml 的 SPRITE 指向 {(out / 'manifest.json').as_posix()}")


if __name__ == "__main__":
    make_demo(sys.argv[1] if len(sys.argv) > 1 else "assets/live2d/demo")
