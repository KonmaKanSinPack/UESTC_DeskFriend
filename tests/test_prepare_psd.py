"""prepare_psd 纯函数单测：左右切分与边缘扩展（合成数组，不需要真 PSD）。"""

import numpy as np

from tools.prepare_psd import edge_extend, split_lr


def _solid(w, h, x0, y0):
    """画布 30x20 上一个实心块（白色不透明）。"""
    a = np.zeros((20, 30, 4), dtype=np.uint8)
    a[y0 : y0 + h, x0 : x0 + w] = (255, 255, 255, 255)
    return a


class TestSplitLr:
    def test_two_blobs_split_at_gap(self):
        arr = np.zeros((10, 30, 4), dtype=np.uint8)
        arr[:, 5:10] = (255, 0, 0, 255)  # 左块
        arr[:, 20:25] = (0, 0, 255, 255)  # 右块
        result = split_lr(arr)
        assert result is not None
        left, right, cut = result
        assert left.shape[1] + right.shape[1] == 30
        assert 10 <= cut <= 20  # 切口落在间隙里
        # 左块只含红、右块只含蓝（各自完整保留、互不混入）
        assert (left[..., 0] == 255).any() and (left[..., 2] == 255).sum() == 0
        assert (right[..., 2] == 255).any() and (right[..., 0] == 255).sum() == 0

    def test_single_blob_returns_none(self):
        arr = _solid(10, 10, 10, 5)
        assert split_lr(arr) is None

    def test_empty_returns_none(self):
        assert split_lr(np.zeros((10, 10, 4), dtype=np.uint8)) is None


class TestEdgeExtend:
    def test_grows_by_px_and_keeps_color(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        arr[5:15, 5:15] = (200, 100, 50, 255)  # 中心 10x10 红块
        out = edge_extend(arr, px=3)
        # 中心颜色不变
        assert tuple(out[3 + 10, 3 + 10]) == (200, 100, 50, 255)
        # 边缘外 2px（仍在扩展范围内）已变实且颜色来自原边缘
        assert out[3 + 10, 3 + 15 + 2, 3] > 0  # 原右缘外 2px 有内容（padding=3）
        assert tuple(out[3 + 10, 3 + 15 + 2][:3]) == (200, 100, 50)

    def test_output_size_padded(self):
        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        arr[3:7, 3:7] = (1, 2, 3, 255)
        out = edge_extend(arr, px=4)
        assert out.shape[:2] == (18, 18)  # 四周各加 4

    def test_far_region_stays_empty(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        arr[5:15, 5:15] = (9, 9, 9, 255)
        out = edge_extend(arr, px=2)
        # 距内容超过 px 的角落仍为空（扩展是有界的）
        assert out[0, 0, 3] == 0
        assert out[-1, -1, 3] == 0
