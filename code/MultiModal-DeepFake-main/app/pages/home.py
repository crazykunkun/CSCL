from __future__ import annotations

import streamlit as st

from app.utils.file_utils import ASSETS_DIR, read_text


def render():
    st.title("基于频域增强的多模态媒体篡改检测系统")
    st.caption("Frequency-Guided CSCL for Detecting and Grounding Multi-Modal Media Manipulation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("基础模型", "CSCL")
    col2.metric("频域特征", "4 类")
    col3.metric("定位任务", "BBox + Token")
    col4.metric("测试集", "DGM4")

    st.subheader("项目简介")
    st.write(
        "本系统基于 CSCL 多模态篡改检测模型，引入频域流增强结构，"
        "利用 SRM、DCT、FFT 和 Haar 小波高频响应补充 RGB 语义特征，"
        "用于提升模型对局部伪造痕迹和图像篡改区域的感知能力。"
    )

    st.subheader("核心改进")
    st.markdown(
        """
        - **多源频域特征提取**：提取 SRM 残差、DCT 高频能量、FFT 高通响应和 Haar 小波高频响应。
        - **轻量 CNN 频域编码器**：将 8 通道频域图编码为与图像 patch 对齐的频域 token。
        - **RGB-频域门控融合**：通过 Cross-Attention 和 Gate 自适应注入频域信息。
        - **多任务联合学习**：保留真假分类、篡改类型识别、bbox 定位和文本 token 定位任务。
        """
    )

    arch = ASSETS_DIR / "frequency_guided_cscl_arch.svg"
    if arch.exists():
        st.subheader("模型结构图")
        st.image(str(arch), use_container_width=True)

    with st.expander("项目进展摘要"):
        text = read_text("project_progress_summary.md", default="未找到 project_progress_summary.md")
        st.markdown(text[:6000])
