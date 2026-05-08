from __future__ import annotations

import streamlit as st

from app.services.inference_service import checkpoint_exists, demo_inference
from app.utils.image_utils import draw_boxes, open_image
from app.utils.text_utils import highlight_tokens


def render():
    st.title("单样本检测")
    st.write("输入图像和文本，展示真假检测、篡改类型、图像定位和文本 token 定位结果。")

    with st.sidebar.expander("模型设置", expanded=True):
        model_type = st.selectbox("模型类型", ["频域增强 CSCL", "原始 CSCL"])
        checkpoint = st.text_input("checkpoint", "/root/autodl-tmp/model/checkpoint_best.pth")
        threshold = st.slider("分类阈值", 0.0, 1.0, 0.5, 0.01)
        use_demo = st.checkbox("使用演示输出", value=True)

    col_input, col_output = st.columns([1, 1])
    with col_input:
        st.subheader("输入")
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])
        text = st.text_area("输入文本", "Gordon Brown is forced to resign EU meeting by Nicolas Sarkozy in Paris.", height=120)
        run = st.button("开始检测", type="primary")

    with col_output:
        st.subheader("输出")
        if not run:
            st.info("上传样本并点击开始检测。")
            return
        image = open_image(uploaded) if uploaded else None
        if not use_demo and not checkpoint_exists(checkpoint):
            st.error("checkpoint 不存在。请检查路径，或勾选使用演示输出。")
            return
        result = demo_inference(text, image.size if image else None)
        st.metric("真假检测", result.binary_label, f"score={result.binary_score:.3f}")
        st.write("篡改类型：", ", ".join(result.labels))
        st.caption(f"模型：{model_type}；阈值：{threshold:.2f}")
        if image:
            st.image(draw_boxes(image, result.boxes, ["Pred"]), caption="预测 bbox", use_container_width=True)
        st.markdown("文本定位：", unsafe_allow_html=True)
        st.markdown(highlight_tokens(text, result.token_positions), unsafe_allow_html=True)
        st.info(result.note)
