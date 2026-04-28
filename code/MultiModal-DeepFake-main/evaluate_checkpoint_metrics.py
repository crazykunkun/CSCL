import argparse
import json
from pathlib import Path

import torch
import yaml
from transformers import RobertaTokenizerFast

from dataset import create_dataset, create_loader
from models.CSCL import CSCL
from test import evaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_2_0.yaml")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/model/checkpoint_49.pth")
    parser.add_argument("--text_encoder", default="/root/autodl-tmp/model/roberta-base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size_val", type=int, default=None)
    parser.add_argument("--output", default="results/eval_checkpoint49_test2_0.json")
    parser.add_argument("--log", action="store_true")
    return parser.parse_args()


def build_model(args, config, device):
    model = CSCL(args=args, config=config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    if args.log:
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


def main():
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if args.batch_size_val is not None:
        config["batch_size_val"] = args.batch_size_val

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder)
    model = build_model(args, config, device)
    val_loader = build_loader(config)

    metrics = evaluation(args, model, val_loader, tokenizer, device, config)
    (
        auc_cls,
        acc_cls,
        eer_cls,
        map_score,
        op,
        or_,
        of1,
        cp,
        cr,
        cf1,
        f1_multicls,
        iou_score,
        iou_acc_50,
        iou_acc_75,
        iou_acc_95,
        acc_tok,
        precision_tok,
        recall_tok,
        f1_tok,
    ) = metrics

    result = {
        "dataset": config["val_file"][0],
        "checkpoint": args.checkpoint,
        "AUC_cls": round(auc_cls * 100, 4),
        "ACC_cls": round(acc_cls * 100, 4),
        "EER_cls": round(eer_cls * 100, 4),
        "MAP": round(map_score * 100, 4),
        "OP": round(op * 100, 4),
        "OR": round(or_ * 100, 4),
        "OF1": round(of1 * 100, 4),
        "CP": round(cp * 100, 4),
        "CR": round(cr * 100, 4),
        "CF1": round(cf1 * 100, 4),
        "F1_FS": round(float(f1_multicls[0]) * 100, 4),
        "F1_FA": round(float(f1_multicls[1]) * 100, 4),
        "F1_TS": round(float(f1_multicls[2]) * 100, 4),
        "F1_TA": round(float(f1_multicls[3]) * 100, 4),
        "IOU_score": round(iou_score * 100, 4),
        "IOU_ACC_50": round(iou_acc_50 * 100, 4),
        "IOU_ACC_75": round(iou_acc_75 * 100, 4),
        "IOU_ACC_95": round(iou_acc_95 * 100, 4),
        "ACC_tok": round(acc_tok * 100, 4),
        "Precision_tok": round(precision_tok * 100, 4),
        "Recall_tok": round(recall_tok * 100, 4),
        "F1_tok": round(f1_tok * 100, 4),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"saved_to={output_path}", flush=True)


if __name__ == "__main__":
    main()
