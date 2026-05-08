from __future__ import annotations

import json
from collections import Counter

import pandas as pd
import streamlit as st

from app.utils.file_utils import read_json, resolve_path
from app.utils.text_utils import highlight_tokens

DEFAULT_CANDIDATES = "results/ambiguous_text_tamper_candidates.json"


def _load_candidates(path: str) -> list[dict]:
    data = read_json(path, default={})
    if isinstance(data, dict):
        return data.get("candidates", []) or []
    if isinstance(data, list):
        return data
    return []


def render():
    st.title("模糊文本篡改样本分析")
    st.write("用于查看 text_swap 与 text_attribute 边界模糊的候选样本。")

    path = st.sidebar.text_input("候选 JSON", DEFAULT_CANDIDATES)
    rows = _load_candidates(path)
    if not rows:
        st.warning("未找到候选样本。")
        return

    st.metric("候选总数", len(rows))
    label_counter = Counter(str(r.get("fake_cls", "")) for r in rows)
    reason_counter = Counter(reason for r in rows for reason in r.get("suspect_reasons", []))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("标签分布")
        st.dataframe(pd.DataFrame(label_counter.items(), columns=["fake_cls", "count"]), use_container_width=True)
    with col2:
        st.subheader("可疑原因分布")
        st.dataframe(pd.DataFrame(reason_counter.items(), columns=["reason", "count"]), use_container_width=True)

    labels = ["全部"] + sorted(label_counter)
    selected_label = st.selectbox("按标签筛选", labels)
    reason_options = ["全部"] + sorted(reason_counter)
    selected_reason = st.selectbox("按可疑原因筛选", reason_options)

    filtered = rows
    if selected_label != "全部":
        filtered = [r for r in filtered if str(r.get("fake_cls", "")) == selected_label]
    if selected_reason != "全部":
        filtered = [r for r in filtered if selected_reason in r.get("suspect_reasons", [])]

    st.subheader(f"样本列表：{len(filtered)} 条")
    table_rows = []
    for r in filtered:
        table_rows.append({
            "id": r.get("id"),
            "fake_cls": r.get("fake_cls"),
            "tamper_ratio": r.get("tamper_ratio"),
            "text_type_margin": r.get("text_type_margin"),
            "tampered_words": " ".join(map(str, r.get("tampered_words", []))),
            "image": r.get("image"),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=320)

    if filtered:
        st.subheader("样本详情")
        idx = st.number_input("样本序号", 0, max(0, len(filtered) - 1), 0)
        sample = filtered[int(idx)]
        st.json({k: sample.get(k) for k in ["id", "image", "fake_cls", "tamper_ratio", "text_type_margin", "suspect_reasons", "tampered_words"]})
        st.markdown(highlight_tokens(sample.get("text", ""), sample.get("fake_text_pos", [])), unsafe_allow_html=True)
        st.download_button(
            "导出当前筛选 JSON",
            data=json.dumps(filtered, ensure_ascii=False, indent=2),
            file_name="ambiguous_filtered.json",
            mime="application/json",
        )
