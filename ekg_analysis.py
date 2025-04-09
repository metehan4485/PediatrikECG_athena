import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def analyze_ekg(pil_image):
    # PIL → OpenCV dönüşümü
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Grid kalibrasyonu için varsayım: her büyük kare 5 mm, her küçük kare 1 mm
    # 25 mm/s hız, 10 mm/mV kalibrasyona göre 1 küçük kare: 0.04 sn, 0.1 mV

    # Kenar bulma ve yatay çizgileri analiz ederek grid yüksekliği çıkar
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=60, maxLineGap=5)

    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            if abs(x1 - x2) < 3:  # Dikey çizgi
                vertical_lines.append((x1, y1, x2, y2))

    vertical_lines = sorted(vertical_lines, key=lambda x: x[0])
    x_coords = [x1 for x1,_,_,_ in vertical_lines]
    diffs = np.diff(x_coords)
    median_spacing = np.median(diffs) if len(diffs) > 0 else 10
    time_per_pixel = 0.04 / median_spacing  # saniye/piksel

    # Sinyal çıkarımı (ortalama yatay çizgi üzerinden)
    y = gray.shape[0] // 2
    signal = gray[y-5:y+5, :]
    trace = 255 - np.mean(signal, axis=0)

    # Basit tepe noktası (R-dalgası) tespiti
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(trace, distance=30, prominence=20)

    # RR aralıkları ve kalp hızı
    rr_intervals = np.diff(peaks) * time_per_pixel  # saniye
    if len(rr_intervals) > 0:
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr
    else:
        heart_rate = 0

    # Görsel üzerine çizim
    pil_result = pil_image.copy()
    draw = ImageDraw.Draw(pil_result)
    for px in peaks:
        draw.ellipse((px-4, y-20, px+4, y-12), outline="red", width=2)
    draw.text((10, 10), f"Heart Rate: {int(heart_rate)} bpm", fill="blue")

    yorum = f"Kalp hızı yaklaşık {int(heart_rate)} bpm olarak ölçüldü. {len(peaks)} R-dalgası tespit edildi."

    return pil_result, yorum
