from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pages import (
    home,
    single_inference,
    frequency_visualization,
    metrics_compare,
    ambiguous_samples,
    training_logs,
)

st.set_page_config(
    page_title="Frequency-Guided CSCL",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "首页 / 项目介绍": home.render,
    "单样本检测": single_inference.render,
    "频域流可视化": frequency_visualization.render,
    "模型指标对比": metrics_compare.render,
    "模糊样本分析": ambiguous_samples.render,
    "训练日志查看": training_logs.render,
}

st.sidebar.title("Frequency-Guided CSCL")
st.sidebar.caption("多模态媒体篡改检测与定位系统")
page_name = st.sidebar.radio("系统导航", list(PAGES.keys()))
st.sidebar.divider()
st.sidebar.subheader("默认路径")
st.sidebar.code("/root/autodl-tmp/CSCL/code/MultiModal-DeepFake-main", language="text")
st.sidebar.caption("如在其他环境运行，请在各页面修改路径。")

PAGES[page_name]()
