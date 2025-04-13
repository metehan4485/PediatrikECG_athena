import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ekg_analysis import (
    calculate_rr_interval,
    calculate_qt_interval,
    calculate_qtc,
    annotate_image
)

st.set_page_config(page_title="Athena – Pediatric ECG", layout="centered")

st.title("🩺 Athena – Pediatric ECG Assistant")
st.markdown("💙 **Hoş geldiniz Dr. Mete.** 10 saniyelik ritm şeridi içeren bir EKG görseli yükleyin. Athena sizin için yorumlasın.")

# 1. Upload ECG image
uploaded_file = st.file_uploader("📤 Upload ECG Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_width = image_np.shape[1]

    # 🧠 Automatic time calibration: 10 seconds = 10,000 ms
    time_scale = 10000 / image_width  # ms per pixel

    st.success(f"🧮 Time scale automatically set: {time_scale:.2f} ms/pixel")
    st.image(image_np, caption="EKG Input – 10 second standard strip", use_column_width=True)

    st.markdown("### ✍️ Mark ECG Points (x-coordinates)")
    qrs_start = st.number_input("QRS Start (x)", 0, image_width, 130)
    r1 = st.number_input("First R Peak (x)", 0, image_width, 160)
    r2 = st.number_input("Second R Peak (x)", 0, image_width, 260)
    t_end = st.number_input("T Wave End (x)", 0, image_width, 310)

    if st.button("📏 Analyze"):
        rr = calculate_rr_interval([r1, r2], time_scale)
        qt = calculate_qt_interval(qrs_start, t_end, time_scale)
        qtc = calculate_qtc(qt, rr)
        heart_rate = (60000 / rr) if rr else None

        result_img = annotate_image(image_np, rr, qt, qtc)
        st.image(result_img, caption="🖼️ ECG with Measurements", use_column_width=True)

        st.markdown("### 📊 Results")
        st.write(f"🫀 Heart Rate: **{heart_rate:.1f} bpm**" if heart_rate else "🫀 Heart Rate: --")
        st.write(f"📏 RR Interval: **{rr:.1f} ms**" if rr else "RR Interval: --")
        st.write(f"📏 QT Interval: **{qt:.1f} ms**" if qt else "QT Interval: --")
        st.write(f"🧠 QTc (Bazett): **{qtc:.1f} ms**" if qtc else "QTc: --")

        st.success("✅ ECG analysis complete.")
