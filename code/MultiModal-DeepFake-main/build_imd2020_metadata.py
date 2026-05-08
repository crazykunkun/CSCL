import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert IMD2020 to DGM4-style metadata.")
    parser.add_argument("--root", default="/root/autodl-tmp/datasets/forensics/IMD2020")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/datasets/forensics/IMD2020/metadata")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--text", default="This image may contain manipulated visual content.")
    return parser.parse_args()


def rel_to_dataset(path: Path) -> str:
    dataset_root = Path("/root/autodl-tmp/datasets")
    return str(path.resolve().relative_to(dataset_root))


def bbox_from_mask(mask_path: Path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def collect_fake(root: Path):
    img_dir = root / "img"
    mask_dir = root / "mask"
    rows = []
    image_files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for idx, image_path in enumerate(image_files):
        stem = image_path.stem
        candidates = sorted(mask_dir.glob(f"{stem}*_mask.*"))
        if not candidates:
            candidates = sorted(mask_dir.glob(f"{stem.replace('_fake', '')}*_mask.*"))
        mask_path = candidates[0] if candidates else None
        bbox = bbox_from_mask(mask_path) if mask_path else [0, 0, 0, 0]
        rows.append({
            "id": f"imd2020_fake_{idx:06d}",
            "image": rel_to_dataset(image_path),
            "fake_cls": "face_attribute",
            "text": "This image contains manipulated visual content.",
            "fake_text_pos": [],
            "fake_image_box": bbox,
            "mask": rel_to_dataset(mask_path) if mask_path else "",
            "source_dataset": "IMD2020",
        })
    return rows


def collect_authentic(root: Path):
    authentic_dir = root / "Authentic"
    rows = []
    image_files = sorted(p for p in authentic_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for idx, image_path in enumerate(image_files):
        rows.append({
            "id": f"imd2020_auth_{idx:06d}",
            "image": rel_to_dataset(image_path),
            "fake_cls": "orig",
            "text": "This image is authentic visual content.",
            "fake_text_pos": [],
            "fake_image_box": [0, 0, 0, 0],
            "mask": "",
            "source_dataset": "IMD2020",
        })
    return rows


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    rows = collect_fake(root) + collect_authentic(root)
    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    write_json(out / "imd2020_all.json", rows)
    write_json(out / "imd2020_train.json", train)
    write_json(out / "imd2020_val.json", val)
    write_json(out / "imd2020_test.json", test)

    summary = {
        "root": str(root),
        "total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "fake": sum(r["fake_cls"] != "orig" for r in rows),
        "authentic": sum(r["fake_cls"] == "orig" for r in rows),
        "note": "Converted to DGM4-style metadata. fake_cls uses face_attribute as image-only manipulation placeholder.",
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
