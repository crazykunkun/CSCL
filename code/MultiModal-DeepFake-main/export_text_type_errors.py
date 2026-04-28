import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import RobertaTokenizerFast

from dataset import create_dataset, create_loader
from models.CSCL import CSCL
from test import text_input_adjust


TEXT_LABEL_NAMES = ["text_swap", "text_attribute"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test.yaml")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/model/checkpoint_49.pth")
    parser.add_argument("--text_encoder", default="/root/autodl-tmp/model/roberta-base")
    parser.add_argument("--output", default="results/text_type_errors_checkpoint49.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(args, config):
    model = CSCL(args=args, config=config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def build_loader(config):
    _, val_dataset = create_dataset(config)
    val_loader = create_loader(
        [val_dataset],
        [None],
        batch_size=[config["batch_size_val"]],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None],
    )[0]
    return val_loader


def get_targets(labels):
    targets = np.zeros((len(labels), 2), dtype=np.int64)
    for index, label in enumerate(labels):
        if "text_swap" in label:
            targets[index, 0] = 1
        if "text_attribute" in label:
            targets[index, 1] = 1
    return targets


def get_target_names(target_bits):
    names = [TEXT_LABEL_NAMES[idx] for idx, bit in enumerate(target_bits) if bit == 1]
    return names or ["none"]


@torch.no_grad()
def collect_errors(model, device, loader, tokenizer):
    errors = []
    total_samples = 0
    total_errors = 0

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

        pred_text = (logits_multicls[:, 2:] >= 0).long().cpu().numpy()
        target_text = get_targets(label)

        mismatch_mask = np.any(pred_text != target_text, axis=1)
        total_samples += len(label)
        total_errors += int(mismatch_mask.sum())

        mismatch_indices = np.where(mismatch_mask)[0].tolist()
        for sample_index in mismatch_indices:
            errors.append(
                {
                    "image": image_path[sample_index],
                    "label": label[sample_index],
                    "gt_text_type": get_target_names(target_text[sample_index].tolist()),
                    "pred_text_type": get_target_names(pred_text[sample_index].tolist()),
                    "pred_text_logits": logits_multicls[sample_index, 2:].detach().cpu().tolist(),
                    "batch_index": batch_index,
                    "sample_index_in_batch": sample_index,
                }
            )

    return {
        "total_samples": total_samples,
        "text_type_error_count": total_errors,
        "text_type_error_rate": total_errors / total_samples if total_samples else 0.0,
        "errors": errors,
    }


def main():
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    model, device = build_model(args, config)
    loader = build_loader(config)
    payload = collect_errors(model, device, loader, tokenizer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in payload.items() if k != "errors"}, indent=2))
    print(f"saved_to={output_path}")


if __name__ == "__main__":
    main()
