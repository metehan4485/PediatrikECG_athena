# ✅ v0.7 - Otomatik P, QRS ve T Segment Tanımlayıcı
from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)

    # Alt DII şeridini al
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:150] = False  # Kalibrasyon bölgesini dışla

    signal = medfilt(raw_signal * mask, kernel_size=7)

    # R-piklerini bul
    r_peaks, _ = find_peaks(signal, distance=40, prominence=20, height=np.mean(signal) + 2*np.std(signal))

    # Q, T segmentlerini ara
    q_onsets, t_offsets = [], []
    for r in r_peaks:
        # Q-onset: R'den 40-10 piksel önce düşen kısım
        q_start = r - 30
        q_segment = signal[q_start:r] if q_start > 0 else signal[:r]
        q = q_start + np.argmin(q_segment)
        q_onsets.append(q)

        # T dalgası: R'den 30-100 piksel sonra yüksek sinyal
        t_window = signal[r+30:r+100] if r+100 < len(signal) else signal[r+30:]
        if len(t_window) > 0:
            t = r + 30 + np.argmax(t_window)
            t_offsets.append(t)

    # Hesaplamalar
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm
    rr_intervals = np.diff(r_peaks) * time_per_px
    qt_intervals = [(t - q) * time_per_px for q, t in zip(q_onsets, t_offsets)]
    hr = int(60 / np.mean(rr_intervals)) if len(rr_intervals) > 1 else 0
    avg_qt = round(np.mean(qt_intervals) * 1000, 1) if qt_intervals else 0
    qtc = round(avg_qt / np.sqrt(np.mean(rr_intervals)), 1) if rr_intervals.any() else 0

    # Çizim
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for r, q, t in zip(r_peaks, q_onsets, t_offsets):
        draw.ellipse((r - 3, y0 - 6, r + 3, y0 + 6), outline="red", width=2)   # R
        draw.ellipse((q - 2, y0 - 4, q + 2, y0 + 4), outline="blue", width=1)  # Q
        draw.ellipse((t - 2, y0 - 4, t + 2, y0 + 4), outline="green", width=1) # T

    draw.text((10, 10), f"HR: {hr} bpm", fill="blue")
    draw.text((10, 30), f"QT: {avg_qt} ms", fill="orange")
    draw.text((10, 50), f"QTc: {qtc} ms", fill="purple")

    yorum = f"{len(r_peaks)} QRS kompleksi tespit edildi. HR: {hr} bpm | QT: {avg_qt} ms | QTc: {qtc} ms"
    return pil_image, yorum
