"""番茄图片混淆/解混淆核心算法。

基于广义希尔伯特曲线（Gilbert Curve）的像素循环移位混淆。
参考 xiaofanqie-source/index.html 的 JavaScript 实现，使用 numpy 加速。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image


def _sign(x: int) -> int:
    """等价于 JavaScript 的 Math.sign，对整数返回 -1 / 0 / 1。"""
    return (x > 0) - (x < 0)


def gilbert2d(width: int, height: int) -> List[Tuple[int, int]]:
    """生成 width x height 矩形的 Gilbert 曲线坐标序列。

    返回的列表长度等于 width * height，每个元素为 (x, y)，
    表示曲线上第 i 个位置对应的二维坐标。
    """
    coordinates: List[Tuple[int, int]] = []
    if width >= height:
        _generate2d(0, 0, width, 0, 0, height, coordinates)
    else:
        _generate2d(0, 0, 0, height, width, 0, coordinates)
    return coordinates


def _generate2d(x: int, y: int, ax: int, ay: int, bx: int, by: int,
                coordinates: List[Tuple[int, int]]) -> None:
    w = abs(ax + ay)
    h = abs(bx + by)

    dax = _sign(ax)
    day = _sign(ay)
    dbx = _sign(bx)
    dby = _sign(by)

    if h == 1:
        for _ in range(w):
            coordinates.append((x, y))
            x += dax
            y += day
        return

    if w == 1:
        for _ in range(h):
            coordinates.append((x, y))
            x += dbx
            y += dby
        return

    ax2 = ax // 2
    ay2 = ay // 2
    bx2 = bx // 2
    by2 = by // 2

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2 += dax
            ay2 += day
        _generate2d(x, y, ax2, ay2, bx, by, coordinates)
        _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, coordinates)
    else:
        if (h2 % 2) and (h > 2):
            bx2 += dbx
            by2 += dby
        _generate2d(x, y, bx2, by2, ax2, ay2, coordinates)
        _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2, coordinates)
        _generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2),
            coordinates,
        )


def _curve_indices(width: int, height: int) -> np.ndarray:
    """生成 Gilbert 曲线对应的 1D 像素索引数组（按行存储）。"""
    curve = gilbert2d(width, height)
    return np.fromiter((x + y * width for x, y in curve), dtype=np.int64, count=len(curve))


def _golden_offset(total: int) -> int:
    """黄金分割比偏移量：round((sqrt(5)-1)/2 * total)。"""
    return round((math.sqrt(5) - 1) / 2 * total)


def obfuscate(img: Image.Image) -> Image.Image:
    """对 PIL 图像执行混淆（循环移位）操作，返回新的 PIL 图像。

    算法：将所有像素按 Gilbert 曲线顺序排列，然后整体循环移位 offset 步。
    """
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    width, height = img.size
    total = width * height
    if total <= 1:
        return img.copy()

    pixels = np.asarray(img)
    flat = pixels.reshape(total, -1)

    indices = _curve_indices(width, height)
    offset = _golden_offset(total)

    # 混淆：new[dst] = old[src]，其中 dst = (i + offset) % total
    src = indices
    dst = indices[(np.arange(total) + offset) % total]

    new_flat = np.empty_like(flat)
    new_flat[dst] = flat[src]
    return Image.fromarray(new_flat.reshape(pixels.shape), mode=img.mode)


def deobfuscate(img: Image.Image) -> Image.Image:
    """对 PIL 图像执行解混淆（反向循环移位）操作，返回新的 PIL 图像。

    算法：与 obfuscate 互逆，old[src] = new[dst]。
    """
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    width, height = img.size
    total = width * height
    if total <= 1:
        return img.copy()

    pixels = np.asarray(img)
    flat = pixels.reshape(total, -1)

    indices = _curve_indices(width, height)
    offset = _golden_offset(total)

    src = indices
    dst = indices[(np.arange(total) + offset) % total]

    restored = np.empty_like(flat)
    restored[src] = flat[dst]
    return Image.fromarray(restored.reshape(pixels.shape), mode=img.mode)


def save_jpeg(img: Image.Image, path: str, quality: float = 0.95) -> None:
    """以 JPEG 格式保存图像，quality 取 0.0~1.0（与 JS 的 toBlob quality 一致）。"""
    if img.mode == "RGBA":
        img = img.convert("RGB")
    q = max(1, min(100, int(round(quality * 100))))
    img.save(path, format="JPEG", quality=q, subsampling=0)


def save_png(img: Image.Image, path: str) -> None:
    """以 PNG 格式保存图像（无损），适合需要解混淆还原的场景。"""
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img.save(path, format="PNG")
