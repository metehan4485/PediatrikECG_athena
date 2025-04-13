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
from utils import get_scale_from_points

st.set_page_config(page_title="Athena - Pediatric ECG", layout="centered")

st.title("🩺 Athena – Pediatric ECG Assistant")
st.markdown("💙 **Welcome, Dr. Mete.** Let's interpret pediatric ECGs together. Please upload an image to begin.")

# 1. Upload EKG image
uploaded_file = st.file_uploader("📤 Upload ECG Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="EKG Input", use_column_width=True)

    st.markdown("### 🔧 Calibration")
    st.info("Select two vertical lines 5 small squares apart to calibrate time scale.")

    x1 = st.number_input("First calibration point (x1)", 0, image_np.shape[1], 100)
    x2 = st.number_input("Second calibration point (x2)", 0, image_np.shape[1], 120)

    if st.button("✅ Calibrate"):
        time_scale = get_scale_from_points(x1, x2)
        st.success(f"Calibration complete! Time scale: {time_scale:.2f} ms/pixel")

        st.markdown("### ✍️ Mark ECG Points (x-coordinates)")
        qrs_start = st.number_input("QRS Start (x)", 0, image_np.shape[1], 130)
        r1 = st.number_input("First R Peak (x)", 0, image_np.shape[1], 160)
        r2 = st.number_input("Second R Peak (x)", 0, image_np.shape[1], 260)
        t_end = st.number_input("T Wave End (x)", 0, image_np.shape[1], 310)

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

            st.success("✅ Analysis complete! You can now interpret visually.")
