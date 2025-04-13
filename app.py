import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ekg_segment_analysis import (
    calculate_rr_interval,
    calculate_qt_interval,
    calculate_qtc,
    annotate_image
)
from utils import get_scale_from_points

st.set_page_config(page_title="PediatrikECG_athena", layout="centered")

st.title("🩺 PediatrikECG_athena – Beta")
st.markdown("Arven’e özel 🌸 Hoş geldiniz! Lütfen bir EKG görseli yükleyin ve iki dikey çizgi arası **5 küçük kare** olacak şekilde kalibrasyon noktalarını seçin.")

uploaded_file = st.file_uploader("📤 EKG Görseli Yükle (PNG/JPG)", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="Yüklenen EKG", use_column_width=True)

    st.markdown("### 📐 Grid Kalibrasyonu")
    st.info("🔍 5 küçük karelik iki dikey çizgiyi seçin. Bu sayede zaman kalibrasyonu yapılır.")

    calibration_points = st.experimental_data_editor(
        pd.DataFrame({"x": [0, 100], "y": [100, 100]}),
        use_container_width=True,
        num_rows="fixed"
    )

    if st.button("✅ Kalibrasyonu Başlat"):
        x1, x2 = calibration_points["x"]
        pixels_per_5mm = abs(x2 - x1)
        mm_per_ms = 0.2  # 25 mm/s
        time_scale = (5 / pixels_per_5mm) / mm_per_ms * 1000  # ms per pixel

        st.success(f"Zaman kalibrasyonu tamamlandı ✅ Ölçek: {time_scale:.2f} ms/pixel")

        st.markdown("### 🖱️ Ölçüm Noktalarını Seçin")
        st.info("Sırasıyla: QRS start, QRS end, R1, R2, T end noktalarını girin (x koordinatları)")

        qrs_start = st.number_input("QRS Start (x)", 0, image_np.shape[1], 100)
        qrs_end = st.number_input("QRS End (x)", 0, image_np.shape[1], 120)
        r1 = st.number_input("R1 (x)", 0, image_np.shape[1], 150)
        r2 = st.number_input("R2 (x)", 0, image_np.shape[1], 250)
        t_end = st.number_input("T End (x)", 0, image_np.shape[1], 300)

        if st.button("📏 Ölçümleri Hesapla"):
            rr = calculate_rr_interval([r1, r2], time_scale)
            qt = calculate_qt_interval(qrs_start, t_end, time_scale)
            qtc = calculate_qtc(qt, rr)

            result_img = annotate_image(image_np, rr, qt, qtc)
            st.image(result_img, caption="🧠 Ölçümlü EKG", use_column_width=True)

            st.success("🎉 Ölçüm tamamlandı! Yukarıdaki görselde sonuçları görebilirsiniz.")
