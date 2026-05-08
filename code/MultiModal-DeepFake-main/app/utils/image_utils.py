from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def open_image(path_or_file) -> Image.Image | None:
    try:
        return Image.open(path_or_file).convert("RGB")
    except Exception:
        return None


def draw_boxes(image: Image.Image, boxes: Iterable[Iterable[float]], labels: Iterable[str] | None = None) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    labels = list(labels or [])
    width, height = out.size
    for idx, box in enumerate(boxes):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        if max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        if idx < len(labels):
            draw.text((x1 + 3, y1 + 3), labels[idx], fill="red")
    return out


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - np.nanmin(array)
    denom = np.nanmax(array) + 1e-8
    return np.uint8(np.clip(array / denom, 0, 1) * 255)
