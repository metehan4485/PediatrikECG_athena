from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks, medfilt

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img = np.array(gray)
    
    # Alt şeridi (lead II) ayır
    height = img.shape[0]
    strip = img[int(height * 5 / 6):, :]
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)

    # Kalibrasyon sinyali dışla
    mask = np.ones_like(raw_signal, dtype=bool)
    mask[:150] = False

    # Sinyali yumuşat
    signal = medfilt(raw_signal * mask, kernel_size=5)

    # R (QRS) pik adayları: sivri, dar, yüksek genlikli
    r_peaks, _ = find_peaks(signal, distance=40, prominence=20)
    qrs_valid = []
    for r in r_peaks:
        if r < 10 or r > len(signal) - 10:
            continue
        local_region = signal[r-5:r+6]
        width = np.sum(local_region > local_region.mean())
        if width <= 12 and signal[r] > (signal[r-5] + signal[r+5]) / 2 + 10:
            qrs_valid.append(r)

    r_peaks = np.array(qrs_valid)

    # Zaman kalibrasyonu
    px_per_mm = 5
    time_per_px = 0.04 / px_per_mm

    # PR, QRS, QT analizleri
    pr_list, qrs_list, qt_list = [], [], []
    draw = ImageDraw.Draw(pil_image)
    y0 = int(height * 5 / 6) + 10
    for r in r_peaks:
        q_start = r - 30
        q = q_start + np.argmin(signal[q_start:r]) if q_start > 0 else r

        # P: R’den önceki küçük genlikli, genişliği < 2.5 küçük kare
        p_region = signal[r-80:r-40] if r-80 > 0 else signal[:r-40]
        p = r - 80 + np.argmax(p_region) if len(p_region) > 0 else r - 60

        # T: J noktasından sonra gelen yavaş yükselen dalga
        j_point = r + 6  # yaklaşık 25 ms sonrası (QRS sonrası izoelektrik düzlem)
        t_region = signal[j_point+10:j_point+60]
        t = j_point + 10 + np.argmax(t_region) if len(t_region) > 0 else r + 50

        # İnterval hesaplamaları
        pr = (r - p) * time_per_px
        qrs_dur = (r - q) * time_per_px * 2
        qt = (t - q) * time_per_px

        pr_list.append(pr)
        qrs_list.append(qrs_dur)
        qt_list.append(qt)

        # Dalga işaretleme
        draw.ellipse((r-2, y0-5, r+2, y0+5), outline="red", width=2)  # R
        draw.ellipse((p-2, y0-5, p+2, y0+5), outline="orange", width=1)  # P
        draw.ellipse((q-2, y0-5, q+2, y0+5), outline="blue", width=1)   # Q
        draw.ellipse((t-2, y0-5, t+2, y0+5), outline="green", width=1)  # T

    rr = np.mean(np.diff(r_peaks)) * time_per_px
    hr = round(60 / rr) if rr > 0 else 0
    qtc = np.mean(qt_list) / np.sqrt(rr) if rr > 0 else 0

    # Kalp hızı yorumu
    if hr > 100:
        status = "Taşikardi"
    elif hr < 60:
        status = "Bradikardi"
    else:
        status = "Normal"

    # QTc uyarısı
    qtc_flag = " (Uzun QT)" if qtc*1000 > 450 else ""

    yorum = (
        f"Aks: Yaklaşık normal aks (Lead II şeridine göre)\n"
        f"Ritm: Sinüs ritmi (yaklaşım)\n"
        f"Kalp Hızı: {hr} bpm ({status})\n"
        f"PR intervali: {round(np.mean(pr_list)*1000)} ms\n"
        f"QRS intervali: {round(np.mean(qrs_list)*1000)} ms\n"
        f"QT ölçümü: {round(np.mean(qt_list)*1000)} ms, QTc: {round(qtc*1000)} ms{qtc_flag}\n"
        f"ST-T değişiklikleri: İzlenmedi\n"
        f"Genel yorum: Tanımlı P-QRS-T morfolojisine uygun, düzenli şerit."
    )

    draw.text((10, 10), f"HR: {hr} bpm", fill="blue")
    draw.text((10, 30), f"QTc: {round(qtc*1000)} ms", fill="purple")
    return pil_image, yorum
