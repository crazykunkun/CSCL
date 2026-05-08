from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils.file_utils import read_text, resolve_path

DEFAULT_LOG_DIR = "results/logfrequency_test2_6000_10epoch_sim5_freq1"


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return pd.DataFrame(rows)


def render():
    st.title("训练日志查看")
    log_dir = resolve_path(st.sidebar.text_input("日志目录", DEFAULT_LOG_DIR))
    st.write(f"当前日志目录：`{log_dir}`")

    shell_text = read_text(log_dir / "shell.txt", default="")
    log_df = _read_jsonl(log_dir / "log.txt")

    if log_df.empty and not shell_text:
        st.warning("未找到 shell.txt 或 log.txt。")
        return

    if not log_df.empty:
        st.subheader("结构化日志")
        st.dataframe(log_df, use_container_width=True)
        numeric_cols = [c for c in log_df.columns if pd.api.types.is_numeric_dtype(log_df[c])]
        preferred = [c for c in numeric_cols if any(k in c.lower() for k in ["loss", "iou", "map", "f1", "auc", "acc"])]
        if preferred:
            st.subheader("训练/验证曲线")
            chart_cols = st.multiselect("选择曲线", preferred, default=preferred[: min(5, len(preferred))])
            if chart_cols:
                st.line_chart(log_df[chart_cols])

    if shell_text:
        st.subheader("shell.txt")
        tail_lines = st.slider("显示末尾行数", 50, 1000, 200, 50)
        lines = shell_text.splitlines()[-tail_lines:]
        st.code("\n".join(lines), language="text")
