from __future__ import annotations

import pandas as pd

from app.utils.file_utils import read_json

DEFAULT_SUMMARY = "results/result_summary_test3_0.json"
DEFAULT_OLD = "results/eval_checkpoint49_test3_0.json"
DEFAULT_NEW = "results/eval_frequency_test2_6000_10epoch_sim5_freq1_best_test3_0_bs64.json"


def load_comparison(summary_path: str = DEFAULT_SUMMARY, old_path: str = DEFAULT_OLD, new_path: str = DEFAULT_NEW) -> pd.DataFrame:
    summary = read_json(summary_path, default={})
    if isinstance(summary, dict) and summary.get("comparison"):
        return pd.DataFrame(summary["comparison"])

    old = read_json(old_path, default={}) or {}
    new = read_json(new_path, default={}) or {}
    keys = [k for k in old.keys() if k not in {"dataset", "checkpoint"} and isinstance(old.get(k), (int, float))]
    rows = []
    for key in keys:
        if key in new and isinstance(new[key], (int, float)):
            rows.append({"metric": key, "original": old[key], "new": new[key], "delta": round(new[key] - old[key], 4)})
    return pd.DataFrame(rows)
