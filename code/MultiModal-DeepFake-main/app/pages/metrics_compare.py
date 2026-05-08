from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services.metrics_service import load_comparison, DEFAULT_SUMMARY, DEFAULT_OLD, DEFAULT_NEW
from app.utils.plot_utils import style_delta_dataframe


def render():
    st.title("模型指标对比")
    st.write("对比原始 CSCL 与频域增强 CSCL 在指定测试集上的指标表现。")

    with st.sidebar.expander("指标文件", expanded=False):
        summary_path = st.text_input("summary JSON", DEFAULT_SUMMARY)
        old_path = st.text_input("原模型 JSON", DEFAULT_OLD)
        new_path = st.text_input("新模型 JSON", DEFAULT_NEW)

    df = load_comparison(summary_path, old_path, new_path)
    if df.empty:
        st.warning("未找到可展示的指标文件。")
        return

    st.subheader("指标表")
    st.dataframe(style_delta_dataframe(df), use_container_width=True)

    chart_df = df.copy()
    if {"metric", "original", "new"}.issubset(chart_df.columns):
        chart_df = chart_df.set_index("metric")[["original", "new"]]
        st.subheader("原模型 vs 新模型")
        st.bar_chart(chart_df)

    if "delta" in df.columns:
        st.subheader("指标变化")
        delta_df = df.set_index("metric")[["delta"]]
        st.bar_chart(delta_df)

    st.caption("绿色代表提升，红色代表下降。具体结论需结合测试集、checkpoint 和评测脚本共同判断。")
