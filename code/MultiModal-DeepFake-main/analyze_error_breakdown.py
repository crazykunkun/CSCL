import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from dataset import create_dataset, create_loader
from models import box_ops
from models.CSCL import CSCL
from test import text_input_adjust
from tools.multilabel_metrics import get_multi_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/model/checkpoint_49.pth")
    parser.add_argument("--text_encoder", default="/root/autodl-tmp/model/roberta-base")
    parser.add_argument("--output", default="results/error_breakdown_checkpoint49_test.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--bbox_iou_threshold", type=float, default=0.5)
    parser.add_argument("--token_iou_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--save_details", action="store_true")
    return parser.parse_args()


def build_model(args, config, device):
    model_args = SimpleNamespace(
        text_encoder=args.text_encoder,
        device=args.device,
        distributed=False,
        gpu=0,
        rank=0,
        world_size=1,
        log=False,
    )
    model = CSCL(args=model_args, config=config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    message = model.load_state_dict(state_dict, strict=False)
    print(message)
    model.to(device)
    model.eval()
    return model


def build_loader(config, batch_size, num_workers):
    config = dict(config)
    config["batch_size_val"] = batch_size
    _, val_dataset = create_dataset(config)
    return create_loader(
        [val_dataset],
        [None],
        batch_size=[batch_size],
        num_workers=[num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]


def label_groups(labels):
    labels_array = np.array(labels)
    is_real = labels_array == "orig"
    has_image = np.array([("face_swap" in x) or ("face_attribute" in x) for x in labels], dtype=bool)
    has_text = np.array([("text_swap" in x) or ("text_attribute" in x) for x in labels], dtype=bool)
    return is_real, has_image, has_text


def token_iou_and_exact_error(logits_tok, token_label):
    token_pred = logits_tok.argmax(dim=-1)
    valid_mask = token_label != -100
    gt_fake = (token_label == 1) & valid_mask
    pred_fake = (token_pred == 1) & valid_mask

    intersection = (gt_fake & pred_fake).sum(dim=1).float()
    union = (gt_fake | pred_fake).sum(dim=1).float()
    token_iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
    exact_error = ((token_pred != token_label) & valid_mask).any(dim=1)
    return token_iou, exact_error


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


@torch.no_grad()
def analyze(args, model, loader, tokenizer, device):
    counts = {
        "total_samples": 0,
        "total_error_samples": 0,
        "binary_cls_error": 0,
        "image_type_error": 0,
        "text_type_error": 0,
        "bbox_localization_failure_iou50": 0,
        "token_localization_failure_iou50": 0,
        "token_exact_mismatch": 0,
        "image_gt_samples": 0,
        "text_gt_samples": 0,
    }
    by_label = {}
    overlap_counts = {}
    details = []

    for batch_index, (image, label, text, fake_image_box, fake_word_pos, _, _, image_path) in enumerate(tqdm(loader, desc="analyzing")):
        image = image.to(device, non_blocking=True)
        fake_image_box = fake_image_box.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            max_length=128,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        text_input, fake_token_pos, _ = text_input_adjust(text_input, fake_word_pos, device)

        logits_real_fake, logits_multicls, output_coord, logits_tok = model(
            image, label, text_input, fake_image_box, fake_token_pos, is_train=False
        )

        target_multicls, _ = get_multi_label(label, image)
        _, has_image_np, has_text_np = label_groups(label)
        has_image = torch.from_numpy(has_image_np).to(device)
        has_text = torch.from_numpy(has_text_np).to(device)

        cls_label = torch.ones(len(label), dtype=torch.long, device=device)
        cls_label[np.where(np.array(label) == "orig")[0].tolist()] = 0
        binary_error = logits_real_fake.argmax(dim=1) != cls_label

        multicls_pred = (logits_multicls >= 0).long()
        image_type_error = (multicls_pred[:, :2] != target_multicls[:, :2]).any(dim=1)
        text_type_error = (multicls_pred[:, 2:] != target_multicls[:, 2:]).any(dim=1)

        boxes1 = box_ops.box_cxcywh_to_xyxy(output_coord)
        boxes2 = box_ops.box_cxcywh_to_xyxy(fake_image_box)
        bbox_iou, _ = box_ops.box_iou(boxes1, boxes2, test=True)
        bbox_error = (bbox_iou < args.bbox_iou_threshold) & has_image

        token_label = text_input.attention_mask[:, 1:].clone()
        token_label[token_label == 0] = -100
        token_label[token_label == 1] = 0
        for sample_index, fake_pos_sample in enumerate(fake_token_pos):
            for pos in fake_pos_sample:
                if pos < token_label.shape[1]:
                    token_label[sample_index, pos] = 1
        token_iou, token_exact_error = token_iou_and_exact_error(logits_tok, token_label)
        token_iou_error = (token_iou < args.token_iou_threshold) & has_text
        token_exact_error = token_exact_error & has_text

        error_flags = {
            "binary_cls_error": binary_error,
            "image_type_error": image_type_error,
            "text_type_error": text_type_error,
            "bbox_localization_failure_iou50": bbox_error,
            "token_localization_failure_iou50": token_iou_error,
        }
        union_error = torch.zeros(len(label), dtype=torch.bool, device=device)
        for value in error_flags.values():
            union_error |= value

        counts["total_samples"] += len(label)
        counts["total_error_samples"] += int(union_error.sum().item())
        counts["binary_cls_error"] += int(binary_error.sum().item())
        counts["image_type_error"] += int(image_type_error.sum().item())
        counts["text_type_error"] += int(text_type_error.sum().item())
        counts["bbox_localization_failure_iou50"] += int(bbox_error.sum().item())
        counts["token_localization_failure_iou50"] += int(token_iou_error.sum().item())
        counts["token_exact_mismatch"] += int(token_exact_error.sum().item())
        counts["image_gt_samples"] += int(has_image.sum().item())
        counts["text_gt_samples"] += int(has_text.sum().item())

        label_list = list(label)
        union_error_cpu = union_error.cpu().tolist()
        flag_cpu = {name: tensor.cpu().tolist() for name, tensor in error_flags.items()}
        bbox_iou_cpu = bbox_iou.detach().cpu().tolist()
        token_iou_cpu = token_iou.detach().cpu().tolist()

        for sample_index, sample_label in enumerate(label_list):
            entry = by_label.setdefault(
                sample_label,
                {
                    "samples": 0,
                    "total_error_samples": 0,
                    "binary_cls_error": 0,
                    "image_type_error": 0,
                    "text_type_error": 0,
                    "bbox_localization_failure_iou50": 0,
                    "token_localization_failure_iou50": 0,
                },
            )
            entry["samples"] += 1
            entry["total_error_samples"] += int(union_error_cpu[sample_index])
            active_names = []
            for name, values in flag_cpu.items():
                if values[sample_index]:
                    entry[name] += 1
                    active_names.append(name)
            if active_names:
                key = "+".join(active_names)
                overlap_counts[key] = overlap_counts.get(key, 0) + 1
            if args.save_details and union_error_cpu[sample_index]:
                details.append(
                    {
                        "image": image_path[sample_index],
                        "label": sample_label,
                        "errors": active_names,
                        "binary_pred_fake_prob": float(F.softmax(logits_real_fake, dim=1)[sample_index, 1].detach().cpu()),
                        "multicls_logits": logits_multicls[sample_index].detach().cpu().tolist(),
                        "bbox_iou": bbox_iou_cpu[sample_index],
                        "token_iou": token_iou_cpu[sample_index],
                        "batch_index": batch_index,
                        "sample_index_in_batch": sample_index,
                    }
                )

    rates = {
        "total_error_rate": safe_ratio(counts["total_error_samples"], counts["total_samples"]),
        "binary_cls_error_rate": safe_ratio(counts["binary_cls_error"], counts["total_samples"]),
        "image_type_error_rate": safe_ratio(counts["image_type_error"], counts["total_samples"]),
        "text_type_error_rate": safe_ratio(counts["text_type_error"], counts["total_samples"]),
        "bbox_failure_rate_on_image_gt": safe_ratio(counts["bbox_localization_failure_iou50"], counts["image_gt_samples"]),
        "token_failure_rate_on_text_gt": safe_ratio(counts["token_localization_failure_iou50"], counts["text_gt_samples"]),
    }
    return {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "batch_size": args.batch_size,
        "definitions": {
            "binary_cls_error": "argmax(logits_real_fake) != real/fake target",
            "image_type_error": "any mismatch in face_swap/face_attribute multi-label logits using threshold 0",
            "text_type_error": "any mismatch in text_swap/text_attribute multi-label logits using threshold 0",
            "bbox_localization_failure_iou50": "GT image-manipulated samples with bbox IoU < 0.5",
            "token_localization_failure_iou50": "GT text-manipulated samples with fake-token mask IoU < 0.5",
            "token_exact_mismatch": "GT text-manipulated samples with any valid token mismatch; auxiliary strict count",
            "total_error_samples": "sample-level union of binary/image-type/text-type/bbox-iou50/token-iou50 errors",
        },
        "counts": counts,
        "rates": rates,
        "by_label": by_label,
        "overlap_counts": overlap_counts,
        "details": details,
    }


def main():
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        print(f"gpu={props.name}, total_vram_mib={props.total_memory // 1024 // 1024}")
    print(f"batch_size_val={args.batch_size}")

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    model = build_model(args, config, device)
    loader = build_loader(config, args.batch_size, args.num_workers)
    payload = analyze(args, model, loader, tokenizer, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    compact = {k: v for k, v in payload.items() if k not in {"details", "by_label", "overlap_counts"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
