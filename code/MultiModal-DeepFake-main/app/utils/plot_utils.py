from __future__ import annotations

import pandas as pd
import streamlit as st


def metric_card(label: str, value, delta=None):
    st.metric(label, value, delta)


def comparison_dataframe(summary: dict) -> pd.DataFrame:
    rows = summary.get("comparison", []) if isinstance(summary, dict) else []
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def style_delta_dataframe(df: pd.DataFrame):
    if df.empty or "delta" not in df.columns:
        return df
    def color_delta(value):
        try:
            value = float(value)
        except Exception:
            return ""
        if value > 0:
            return "color: #2e7d32; font-weight: 600"
        if value < 0:
            return "color: #c62828; font-weight: 600"
        return "color: #616161"
    return df.style.applymap(color_delta, subset=["delta"])
