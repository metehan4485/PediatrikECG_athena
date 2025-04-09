from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks

def analyze_ekg(pil_image):
    # Görseli griye çevir
    gray_image = pil_image.convert("L")
    img_array = np.array(gray_image)

    # Ortadaki çizgiden sinyal çıkar
    center_y = img_array.shape[0] // 2
    trace_band = img_array[center_y - 5:center_y + 5, :]
    signal = 255 - np.mean(trace_band, axis=0)

    # R-dalgası benzeri pik tespiti
    peaks, _ = find_peaks(signal, distance=30, prominence=20)

    # Kalibrasyon: her küçük kare 1 mm, 25 mm/s → 1 mm = 0.04 sn
    pixel_per_mm = 5  # kabaca tahmini; grid yoksa sabit kabul edilir
    time_per_pixel = 0.04 / pixel_per_mm  # sn/pixel

    # RR hesaplama
    rr_intervals = np.diff(peaks) * time_per_pixel
    if len(rr_intervals) > 0:
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr
    else:
        heart_rate = 0

    # Görsel üzerine işaretleme
    draw = ImageDraw.Draw(pil_image)
    for x in peaks:
        draw.ellipse((x - 4, center_y - 20, x + 4, center_y - 12), outline="red", width=2)
    draw.text((10, 10), f"Heart Rate: {int(heart_rate)} bpm", fill="blue")

    yorum = f"{len(peaks)} adet R-dalgası tespit edildi. Kalp hızı: {int(heart_rate)} bpm."

    return pil_image, yorum
