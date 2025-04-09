from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)

    # Alt uzun DII stripini al (son 1/6)
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]

    # Sinyal çıkar (ortalama çizgi)
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)

    # Kalibrasyon maskesi: ilk 250 pikseli dışla
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:250] = False
    filtered_signal = medfilt(raw_signal, kernel_size=5) * mask

    # Pik tespiti ve genişlik hesabı
    peaks, properties = find_peaks(filtered_signal, distance=40, prominence=20, width=3)
    widths = properties["widths"]
    prominences = properties["prominences"]

    # R / T ayrımı (sivri ve yüksek olanları seç)
    valid_peaks = []
    for i, p in enumerate(peaks):
        sharpness = prominences[i] / widths[i]
        if sharpness > 6:
            valid_peaks.append(p)
    valid_peaks = np.array(valid_peaks)

    # RR ve HR
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm
    rr_intervals = np.diff(valid_peaks) * time_per_px
    hr = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0

    # Çizim
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for x in valid_peaks:
        draw.ellipse((x - 3, y0 - 5, x + 3, y0 + 5), outline="red", width=2)
    draw.text((10, 10), f"Kalp hızı: {hr} bpm", fill="blue")
    draw.text((10, 30), f"RR: {round(np.mean(rr_intervals), 2)} s", fill="green")

    yorum = f"{len(valid_peaks)} QRS kompleksi tespit edildi. Ortalama HR: {hr} bpm."

    return pil_image, yorum
