import streamlit as st
from PIL import Image
from ekg_segment_analysis import analyze_ekg  # Bu fonksiyonu ayrı dosyada yazacağız

st.set_page_config(page_title="PediatrikECG_athena", layout="centered")

# Başlık ve selamlama
st.title("👶 PediatrikECG_athena")
st.markdown("Merhaba **Arven** 🌟")
st.markdown("Bu uygulama, pediatrik EKG görsellerini analiz etmek ve öğretici yorumlar üretmek için tasarlandı.")

# Görsel yükleme
uploaded_file = st.file_uploader("Lütfen bir EKG görseli yükleyin (PNG veya JPG formatında)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Yüklenen EKG", use_container_width=True)

    st.markdown("⏳ Analiz ediliyor...")

    # EKG analizi (analiz fonksiyonu ayrı dosyada olacak)
    result_image, yorum = analyze_ekg(image)

    # Sonuçları göster
    st.image(result_image, caption="📊 Analiz Sonucu", use_container_width=True)
    st.success("🩺 Yorum: " + yorum)
