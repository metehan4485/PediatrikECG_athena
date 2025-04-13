# ✅ v0.6 - Segmentasyon tabanlı EKG analizi (QRS ayrımlı)
from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)

    # Alt DII şeridi al (son 1/6)
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]
    strip_height = strip.shape[0]

    # Kalibrasyon bölgesini maskeden çıkar (ilk 150 piksel)
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:150] = False

    # İlk sinyal düzleme ve filtreleme
    smoothed = medfilt(raw_signal * mask, kernel_size=7)

    # QRS komplekslerini bul: R piki için yeterli sivrilik aranır
    r_peaks, r_prop = find_peaks(smoothed, distance=40, prominence=20)

    # Kalibrasyon değeri (25 mm/s hız, 1 mm = 0.04 s; 5 px = 1 mm kabul)
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm
    rr_intervals = np.diff(r_peaks) * time_per_px
    hr = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 1 else 0

    # Segmentasyon çizimi
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10

    for idx, r in enumerate(r_peaks):
        draw.ellipse((r - 3, y0 - 6, r + 3, y0 + 6), outline="red", width=2)
        if idx < len(rr_intervals):
            draw.text((r + 5, y0 - 15), f"{round(rr_intervals[idx],2)}s", fill="gray")

    draw.text((10, 10), f"Kalp hızı: {hr} bpm", fill="blue")
    draw.text((10, 30), f"RR: {round(np.mean(rr_intervals), 2)} s", fill="green")

    yorum = f"{len(r_peaks)} QRS kompleksi tespit edildi. Ortalama HR: {hr} bpm, RR: {round(np.mean(rr_intervals), 2)} s"
    return pil_image, yorum
