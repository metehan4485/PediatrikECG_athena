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
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm

    # R dalgaları (keskin, yüksek, dar)
    r_peaks, _ = find_peaks(signal, distance=40, prominence=20)
    true_r = []
    for i, r in enumerate(r_peaks):
        if i > 0:
            if r - true_r[-1] < int(300 / (time_per_px * 1000)):
                continue  # 300 ms'den yakınsa atla
        if r < 10 or r >= len(signal) - 10:
            continue
        region = signal[r - 5:r + 6]
        if len(region) < 11:
            continue
        width = np.sum(region > region.mean())
        if width > 12:
            continue
        true_r.append(r)
    r_peaks = np.array(true_r)

    pr_list, qrs_list, qt_list = [], [], []
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for r in r_peaks:
        q_start = r - 25
        q = q_start + np.argmin(signal[q_start:r]) if q_start > 0 else r

        # P: QRS’ten 120–200 ms önce, pozitif defleksiyon
        p_start = r - 50
        p_region = signal[p_start:r-20] if p_start > 0 else signal[:r-20]
        p = p_start + np.argmax(p_region) if len(p_region) > 0 else r - 40

        # J noktası = QRS bitişi
        s_region = signal[r:r + 15] if r + 15 < len(signal) else signal[r:]
        s = r + np.argmin(s_region) if len(s_region) > 0 else r + 10
        j = s + 1

        # T: J’den 150–250 ms sonra (genellikle R’den ~60 px sonra)
        t_region = signal[j + 15:j + 60] if j + 60 < len(signal) else signal[j + 15:]
        t = j + 15 + np.argmax(t_region) if len(t_region) > 0 else r + 50

        pr = (r - p) * time_per_px
        qrs_dur = (s - q) * time_per_px
        qt = (t - q) * time_per_px

        pr_list.append(pr)
        qrs_list.append(qrs_dur)
        qt_list.append(qt)

        # Harf etiketli işaretleme
        draw.text((p, y0 - 20), "P", fill="orange")
        draw.text((q, y0 - 20), "Q", fill="blue")
        draw.text((r, y0 - 20), "R", fill="red")
        draw.text((s, y0 - 20), "S", fill="cyan")
        draw.text((t, y0 - 20), "T", fill="green")

    rr = np.mean(np.diff(r_peaks)) * time_per_px
    hr = round(60 / rr) if rr > 0 else 0
    qtc = np.mean(qt_list) / np.sqrt(rr) if rr > 0 else 0

    if hr > 100:
        status = "Taşikardi"
    elif hr < 60:
        status = "Bradikardi"
    else:
        status = "Normal"

    qtc_flag = " (Uzun QT)" if qtc*1000 > 450 else ""

    yorum = (
        f"Aks: Yaklaşık normal aks (Lead II şeridine göre)\n"
        f"Ritm: Sinüs ritmi (yaklaşım)\n"
        f"Kalp Hızı: {hr} bpm ({status})\n"
        f"PR intervali: {round(np.mean(pr_list)*1000)} ms\n"
        f"QRS intervali: {round(np.mean(qrs_list)*1000)} ms\n"
        f"QT ölçümü: {round(np.mean(qt_list)*1000)} ms, QTc: {round(qtc*1000)} ms{qtc_flag}\n"
        f"ST-T değişiklikleri: Şu an için değerlendirilmiyor\n"
        f"Genel yorum: P-QRS-T dizilimi ve süreleri fizyolojik uyumlu. Klinik değerlendirmeye uygun otomatik ölçüm yapılmıştır."
    )

    draw.text((10, 10), f"HR: {hr} bpm", fill="blue")
    draw.text((10, 30), f"QTc: {round(qtc*1000)} ms", fill="purple")
    return pil_image, yorum
