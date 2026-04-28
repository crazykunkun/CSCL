import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import RobertaTokenizerFast

from dataset import create_dataset, create_loader
from models.CSCL import CSCL
from test import text_input_adjust


TEXT_LABELS = ("text_swap", "text_attribute")
WORD_RE = re.compile(r"\S+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="/root/autodl-tmp/datasets/DGM4/metadata/test.json")
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/model/checkpoint_49.pth")
    parser.add_argument("--text_encoder", default="/root/autodl-tmp/model/roberta-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size_val", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--output_json", default="results/ambiguous_text_tamper_candidates.json")
    parser.add_argument("--output_csv", default="results/ambiguous_text_tamper_candidates.csv")
    parser.add_argument("--no_model", action="store_true")
    return parser.parse_args()


def contains_text_tamper(label):
    return any(name in label for name in TEXT_LABELS)


def text_target_bits(label):
    return {
        "text_swap": int("text_swap" in label),
        "text_attribute": int("text_attribute" in label),
    }


def split_words(text):
    return WORD_RE.findall(text or "")


def tampered_words(text, fake_text_pos):
    words = split_words(text)
    selected = []
    for pos in fake_text_pos or []:
        if isinstance(pos, int) and 0 <= pos < len(words):
            selected.append(words[pos])
    return selected


def longest_consecutive_run(positions):
    if not positions:
        return 0
    positions = sorted(set(int(p) for p in positions))
    best = current = 1
    for prev, cur in zip(positions, positions[1:]):
        if cur == prev + 1:
            current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def entity_like_ratio(words):
    if not words:
        return 0.0
    count = 0
    for word in words:
        stripped = word.strip(".,;:!?()[]{}'\"")
        if not stripped:
            continue
        if any(ch.isdigit() for ch in stripped):
            count += 1
        elif stripped[:1].isupper():
            count += 1
    return count / len(words)


def rule_reasons(label, tamper_ratio, longest_run_ratio, entity_ratio, tampered_count):
    reasons = []
    has_swap = "text_swap" in label
    has_attr = "text_attribute" in label

    if has_swap and tamper_ratio <= 0.15:
        reasons.append("text_swap label but low tamper_ratio, likely local edit")
    if has_swap and entity_ratio >= 0.55 and tampered_count <= 5:
        reasons.append("text_swap label but edited words look like entities/numbers")
    if has_attr and tamper_ratio >= 0.35:
        reasons.append("text_attribute label but high tamper_ratio, likely global replacement")
    if has_attr and longest_run_ratio >= 0.30:
        reasons.append("text_attribute label but long consecutive tampered span")
    if has_swap and has_attr:
        reasons.append("contains both text_swap and text_attribute labels")
    return reasons


def build_rule_candidates(metadata_path):
    data = json.loads(Path(metadata_path).read_text())
    candidates = []
    by_key = {}
    for item in data:
        label = item.get("fake_cls", "")
        if not contains_text_tamper(label):
            continue

        text = item.get("text", "")
        words = split_words(text)
        fake_text_pos = item.get("fake_text_pos") or []
        tampered = tampered_words(text, fake_text_pos)
        word_count = max(len(words), 1)
        tampered_count = len(fake_text_pos)
        tamper_ratio = tampered_count / word_count
        longest_run = longest_consecutive_run(fake_text_pos)
        longest_run_ratio = longest_run / word_count
        ent_ratio = entity_like_ratio(tampered)
        reasons = rule_reasons(label, tamper_ratio, longest_run_ratio, ent_ratio, tampered_count)

        # 规则分数只用于候选排序；后续如有模型分数，会与模型 margin 共同排序。
        score = 0.0
        if reasons:
            score += 1.0
        if "text_swap" in label:
            score += max(0.0, 0.20 - tamper_ratio)
            score += 0.2 * ent_ratio
        if "text_attribute" in label:
            score += max(0.0, tamper_ratio - 0.25)
            score += max(0.0, longest_run_ratio - 0.20)

        key = str(item.get("id", item.get("image")))
        row = {
            "id": item.get("id"),
            "image": item.get("image"),
            "fake_cls": label,
            "text": text,
            "fake_text_pos": fake_text_pos,
            "tampered_words": tampered,
            "word_count": len(words),
            "tampered_count": tampered_count,
            "tamper_ratio": round(tamper_ratio, 6),
            "longest_run": longest_run,
            "longest_run_ratio": round(longest_run_ratio, 6),
            "entity_like_ratio": round(ent_ratio, 6),
            "rule_score": round(score, 6),
            "suspect_reasons": reasons,
            "gt_text_swap": text_target_bits(label)["text_swap"],
            "gt_text_attribute": text_target_bits(label)["text_attribute"],
        }
        candidates.append(row)
        by_key[key] = row
    return candidates, by_key


def normalize_image_key(path):
    if path is None:
        return ""
    path = str(path).replace("\\", "/")
    marker = "DGM4/"
    if marker in path:
        return path[path.index(marker):]
    return path


def build_model(args, config, device):
    model = CSCL(args=args, config=config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg, flush=True)
    model = model.to(device)
    model.eval()
    return model


def build_loader(config):
    _, val_dataset = create_dataset(config)
    return create_loader(
        [val_dataset],
        [None],
        batch_size=[config["batch_size_val"]],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None],
    )[0]


@torch.no_grad()
def attach_model_scores(args, candidates_by_key):
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    config["val_file"] = [args.metadata]
    config["batch_size_val"] = args.batch_size_val

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    model = build_model(args, config, device)
    loader = build_loader(config)

    by_image = {normalize_image_key(row.get("image")): key for key, row in candidates_by_key.items()}

    for batch_index, (image, label, text, fake_image_box, fake_word_pos, _, _, image_path) in enumerate(loader):
        image = image.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)
        _, logits_multicls, _, _ = model(image, label, text_input, fake_image_box, fake_token_pos, is_train=False)
        text_logits = logits_multicls[:, 2:].detach().cpu().float().numpy()
        text_probs = 1.0 / (1.0 + np.exp(-text_logits))

        for idx, path in enumerate(image_path):
            key = by_image.get(normalize_image_key(path))
            if key is None or key not in candidates_by_key:
                continue
            row = candidates_by_key[key]
            swap_logit = float(text_logits[idx, 0])
            attr_logit = float(text_logits[idx, 1])
            swap_prob = float(text_probs[idx, 0])
            attr_prob = float(text_probs[idx, 1])
            margin = abs(swap_logit - attr_logit)
            row.update(
                {
                    "text_swap_logit": round(swap_logit, 6),
                    "text_attribute_logit": round(attr_logit, 6),
                    "text_swap_prob": round(swap_prob, 6),
                    "text_attribute_prob": round(attr_prob, 6),
                    "text_type_margin": round(float(margin), 6),
                    "pred_text_swap": int(swap_logit >= 0),
                    "pred_text_attribute": int(attr_logit >= 0),
                }
            )
            if margin <= 0.2:
                row["suspect_reasons"].append("model text_swap/text_attribute logits are very close")
            elif margin <= 0.5:
                row["suspect_reasons"].append("model text_swap/text_attribute logits are close")

        if batch_index % 100 == 0:
            print(f"processed batch {batch_index}", flush=True)


def sort_key(row):
    margin = row.get("text_type_margin")
    margin_score = 0.0 if margin is None else max(0.0, 1.0 - float(margin))
    reason_score = len(row.get("suspect_reasons", []))
    return (reason_score + margin_score + row.get("rule_score", 0.0), -float(margin or 999.0))


def write_outputs(rows, output_json, output_csv, top_k):
    rows = sorted(rows, key=sort_key, reverse=True)
    if top_k > 0:
        rows = rows[:top_k]

    payload = {
        "description": "Candidates whose text tampering type may be ambiguous between text_swap and text_attribute.",
        "selection_rules": [
            "text_swap with low tamper_ratio",
            "text_swap with entity/number-like edited words",
            "text_attribute with high tamper_ratio",
            "text_attribute with long consecutive tampered span",
            "small model margin between text_swap and text_attribute logits when model scores are available",
        ],
        "count": len(rows),
        "candidates": rows,
    }

    output_json = Path(output_json)
    output_csv = Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    columns = [
        "id",
        "image",
        "fake_cls",
        "tamper_ratio",
        "tampered_count",
        "word_count",
        "longest_run_ratio",
        "entity_like_ratio",
        "text_swap_logit",
        "text_attribute_logit",
        "text_type_margin",
        "text_swap_prob",
        "text_attribute_prob",
        "gt_text_swap",
        "gt_text_attribute",
        "pred_text_swap",
        "pred_text_attribute",
        "suspect_reasons",
        "tampered_words",
        "text",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in columns}
            out["suspect_reasons"] = " | ".join(row.get("suspect_reasons", []))
            out["tampered_words"] = " ".join(row.get("tampered_words", []))
            writer.writerow(out)

    return rows


def main():
    args = parse_args()
    candidates, by_key = build_rule_candidates(args.metadata)
    print(f"metadata text-tamper candidates: {len(candidates)}", flush=True)
    if not args.no_model:
        attach_model_scores(args, by_key)
    rows = write_outputs(candidates, args.output_json, args.output_csv, args.top_k)
    reason_counts = {}
    for row in rows:
        for reason in row.get("suspect_reasons", []):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    print(json.dumps({"saved": [args.output_json, args.output_csv], "count": len(rows), "reason_counts": reason_counts}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
