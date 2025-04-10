import neurokit2 as nk
import numpy as np
from PIL import ImageDraw

def analyze_ekg(pil_image):
    gray = pil_image.convert("L")
    img_array = np.array(gray)

    # Alt şerit (DII): alt 1/6'
    height = img_array.shape[0]
    strip = img_array[int(height * 5 / 6):, :]
    signal_band = strip[5:15, :]
    raw_signal = 255 - np.mean(signal_band, axis=0)

    try:
        cleaned = nk.ecg_clean(raw_signal, sampling_rate=250)
        signals, info = nk.ecg_process(cleaned, sampling_rate=250)

        rpeaks = info['ECG_R_Peaks']
        qonsets = info.get('ECG_Q_Onsets', [])
        tonsets = info.get('ECG_T_Offsets', [])

        qt_intervals = []
        for q, t in zip(qonsets, tonsets):
            qt = (t - q) / 250.0
            qt_intervals.append(qt)

        avg_qt = round(np.mean(qt_intervals) * 1000, 1) if qt_intervals else 0
        avg_rr = round(np.mean(np.diff(rpeaks)) / 250.0, 2) if len(rpeaks) > 1 else 0
        qtc = round(avg_qt / np.sqrt(avg_rr), 1) if avg_rr > 0 else 0

        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 10), f"QT: {avg_qt} ms", fill="blue")
        draw.text((10, 30), f"QTc: {qtc} ms", fill="green")

        yorum = f"QT süre: {avg_qt} ms | QTc: {qtc} ms (Bazett)"

    except Exception as e:
        yorum = f"Segmentasyon hatası: {e}"
        draw = ImageDraw.Draw(pil_image)
        draw.text((10, 10), "❌ Segmentasyon başarısız", fill="red")
        print("Hata detayı: ", e)

    return pil_image, yorum
