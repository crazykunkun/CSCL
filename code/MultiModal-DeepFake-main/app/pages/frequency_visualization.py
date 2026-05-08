from __future__ import annotations

import streamlit as st

from app.services.frequency_service import make_frequency_views
from app.utils.image_utils import open_image


def render():
    st.title("频域流可视化")
    st.write("上传一张图片，查看 SRM、DCT、FFT 和 Haar 四类频域响应的可视化结果。")
    st.info("说明：该页面使用轻量可视化近似实现，用于前端展示；训练时的真实频域算子位于 `models/frequency_branch.py`。")

    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])
    if not uploaded:
        st.warning("请先上传图片。")
        return

    image = open_image(uploaded)
    if image is None:
        st.error("图片读取失败。")
        return

    st.subheader("原图")
    st.image(image, use_container_width=False, width=360)

    views = make_frequency_views(image)
    st.subheader("频域响应")
    cols = st.columns(4)
    for col, (name, view) in zip(cols, views.items()):
        col.image(view, caption=name, use_container_width=True)
