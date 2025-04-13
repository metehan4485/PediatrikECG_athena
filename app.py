import streamlit as st
from PIL import Image
from ekg_analysis import analyze_ekg

st.set_page_config(page_title="PediatrikECG_athena", layout="centered")

st.title("👶 PediatrikECG_athena")
st.markdown("Merhaba **Arven** 🌟")
st.markdown("Bu uygulama, pediatrik EKG görsellerini analiz etmek ve öğretici yorumlar üretmek için tasarlandı.")

uploaded_file = st.file_uploader("EKG görseli yükleyin (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen EKG", use_container_width=True)
    st.markdown("⏳ Otomatik analiz başlıyor...")

    result_image, yorum = analyze_ekg(image)

    st.image(result_image, caption="📊 Analiz Sonucu", use_container_width=True)
    st.success("🩺 **Otomatik Klinik Rapor:**\n\n" + yorum)
