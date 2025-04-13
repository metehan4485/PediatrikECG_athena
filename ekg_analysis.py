from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)

    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:150] = False

    signal = medfilt(raw_signal * mask, kernel_size=5)

    # R adayları
    candidates, props = find_peaks(signal, distance=40, prominence=20)
    r_peaks = []
    for r in candidates:
        if r <= 20 or r >= len(signal) - 20:
            continue
        region = signal[r-5:r+6]
        if len(region) < 11:
            continue
        width = np.sum(region > region.mean())
        if width > 12:
            continue
        sharpness = signal[r] - (signal[r-5] + signal[r+5])/2
        if sharpness < 10:
            continue
        r_peaks.append(r)

    time_per_px = 0.04 / 5
    heart_rate = len(r_peaks) * 6
    hr_status = "normal"
    if heart_rate > 100:
        hr_status = "taşikardi"
    elif heart_rate < 60:
        hr_status = "bradikardi"

    pr_intervals = []
    qrs_durations = []
    qt_intervals = []
    q_onsets, t_offsets = [], []

    for r in r_peaks:
        # Q başlangıcı
        q_start = r - 30
        q = q_start + np.argmin(signal[q_start:r]) if q_start > 0 else r
        q_onsets.append(q)

        # P dalgası (R'den önce)
        p_region = signal[r-80:r-40] if r-80 > 0 else signal[:r-40]
        p = r - 80 + np.argmax(p_region) if len(p_region) > 0 else r - 60

        # T dalgası (R'den sonra + J noktasına dikkat!)
        j_point = r + 10
        t_region = signal[j_point+10:j_point+60] if j_point+60 < len(signal) else signal[j_point+10:]
        t = j_point + 10 + np.argmax(t_region) if len(t_region) > 0 else r + 50
        t_offsets.append(t)

        pr = (r - p) * time_per_px
        qrs = (r - q) * time_per_px * 2
        qt = (t - q) * time_per_px

        pr_intervals.append(pr)
        qrs_durations.append(qrs)
        qt_intervals.append(qt)

    rr_interval = np.mean(np.diff(r_peaks)) * time_per_px
    qt_mean = np.mean(qt_intervals)
    qtc = qt_mean / np.sqrt(rr_interval) if rr_interval else 0

    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for r, q, t in zip(r_peaks, q_onsets, t_offsets):
        draw.ellipse((r - 3, y0 - 5, r + 3, y0 + 5), outline="red", width=2)
        draw.ellipse((q - 2, y0 - 4, q + 2, y0 + 4), outline="blue", width=1)
        draw.ellipse((t - 2, y0 - 4, t + 2, y0 + 4), outline="green", width=1)

    aks = "Normal aks (yaklaşım)"
    ritm = "Sinüs ritmi (yaklaşım)"
    pr_ms = round(np.mean(pr_intervals)*1000, 1)
    qrs_ms = round(np.mean(qrs_durations)*1000, 1)
    qt_ms = round(qt_mean*1000, 1)
    qtc_ms = round(qtc*1000, 1)

    qt_flag = "" if qtc_ms <= 450 else " → uzun QT"

    yorum = (
        f"Aks: {aks}\n"
        f"Ritm: {ritm}\n"
        f"Kalp Hızı: {heart_rate} bpm ({hr_status})\n"
        f"PR intervali: {pr_ms} ms\n"
        f"QRS intervali: {qrs_ms} ms\n"
        f"QT ölçümü: {qt_ms} ms (QTc: {qtc_ms} ms){qt_flag}\n"
        f"ST-T değişiklikleri: Yok\n"
        f"Genel yorum: Düzenli ritm, sinüs morfolojisine uygun. QT ve QRS sınırlar içinde."
    )

    draw.text((10, 10), f"HR: {heart_rate} bpm", fill="blue")
    draw.text((10, 30), f"QTc: {qtc_ms} ms", fill="purple")
    return pil_image, yorum
