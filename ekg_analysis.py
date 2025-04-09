from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    # Görseli gri yap
    gray = pil_image.convert("L")
    img = np.array(gray)

    # Alt şeridi kes (genellikle son 1/6’lık bölge)
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]

    # Sinyal çıkar (yatay kesit ortalaması)
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)

    # Median filtreyle yumuşatma
    smooth_signal = medfilt(raw_signal, kernel_size=5)

    # Pik tespiti (QRS benzeri)
    peaks, _ = find_peaks(smooth_signal, distance=40, prominence=20)

    # Kalibrasyon: 25 mm/s → 1 mm = 0.04 sn; varsayılan 5 px = 1 mm
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm  # saniye/piksel

    rr_intervals = np.diff(peaks) * time_per_px
    hr = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0
    qrs_duration = round(np.mean(rr_intervals) * 1000, 1) if len(rr_intervals) > 0 else 0  # ms

    # Görsel üzerine işaretleme
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for x in peaks:
        draw.ellipse((x - 3, y0 - 5, x + 3, y0 + 5), outline="red", width=2)
    draw.text((10, 10), f"Kalp hızı: {hr} bpm", fill="blue")
    draw.text((10, 30), f"Ortalama RR: {round(np.mean(rr_intervals), 2)} s", fill="green")

    yorum = f"{len(peaks)} QRS kompleksi tespit edildi. Ortalama HR: {hr} bpm, RR: {round(np.mean(rr_intervals),2)} s"

    return pil_image, yorum
