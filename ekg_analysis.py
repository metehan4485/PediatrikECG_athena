from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)

    # Alt uzun DII stripini al (son 1/6)
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]
    strip_height = strip.shape[0]

    # Başlangıçtaki kalibrasyon sinyalini maskele (ilk 150 pikselde sabit tepe)
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:150] = False  # ilk 150 pikseli dışla

    # Filtre uygula
    smooth_signal = medfilt(raw_signal, kernel_size=5)

    # QRS benzeri pikler (R tespiti başlangıcı)
    peaks, properties = find_peaks(smooth_signal * mask, distance=40, prominence=20)

    # T dalgası dışlamaya giriş (yayvan ve düşük eğimli olanları filtrele)
    peak_widths = properties.get("widths", np.ones_like(peaks) * 1.0)
    valid_peaks = []
    for i, p in enumerate(peaks):
        width = peak_widths[i] if i < len(peak_widths) else 5
        height = smooth_signal[p]
        sharpness = height / width
        if sharpness > 10:  # ayarlanabilir eşik
            valid_peaks.append(p)

    valid_peaks = np.array(valid_peaks)

    # Kalibrasyon: 25 mm/s → 1 mm = 0.04 sn; varsayım: 5 px = 1 mm
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm
    rr_intervals = np.diff(valid_peaks) * time_per_px
    hr = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 0 else 0

    # Görsel üzerine işaretleme
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for x in valid_peaks:
        draw.ellipse((x - 3, y0 - 5, x + 3, y0 + 5), outline="red", width=2)
    draw.text((10, 10), f"Kalp hızı: {hr} bpm", fill="blue")
    draw.text((10, 30), f"RR: {round(np.mean(rr_intervals), 2)} s", fill="green")

    yorum = f"{len(valid_peaks)} QRS kompleksi tespit edildi. Ortalama HR: {hr} bpm."

    return pil_image, yorum
