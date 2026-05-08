from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InferenceResult:
    binary_label: str
    binary_score: float
    labels: list[str]
    boxes: list[list[float]]
    token_positions: list[int]
    note: str


def demo_inference(text: str, image_size: tuple[int, int] | None = None) -> InferenceResult:
    words = text.split()
    token_positions = []
    if words:
        token_positions = [min(len(words) - 1, max(0, len(words) // 2))]
    width, height = image_size or (256, 256)
    box = [width * 0.25, height * 0.25, width * 0.75, height * 0.75]
    return InferenceResult(
        binary_label="Fake / 示例输出",
        binary_score=0.873,
        labels=["face_attribute", "text_swap"],
        boxes=[box],
        token_positions=token_positions,
        note="当前前端已预留真实模型推理接口；本页面在未加载 checkpoint 时显示演示结果。",
    )


def checkpoint_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()
