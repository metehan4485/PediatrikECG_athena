import cv2
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

def calculate_rr_interval(r_peaks, time_scale):
    if len(r_peaks) >= 2:
        rr_interval = (r_peaks[1] - r_peaks[0]) * time_scale
        return rr_interval
    else:
        return None

def calculate_qt_interval(qrs_start, t_end, time_scale):
    if qrs_start is not None and t_end is not None:
        qt_interval = abs(t_end - qrs_start) * time_scale
        return qt_interval
    else:
        return None

def calculate_qtc(qt_interval, rr_interval):
    if qt_interval is not None and rr_interval is not None:
        return qt_interval / (rr_interval ** 0.5)
    else:
        return None

def annotate_image(image, rr, qt, qtc):
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 0, 255)
    thickness = 2

    text_lines = [
        f"RR Interval: {rr:.1f} ms" if rr else "RR Interval: --",
        f"QT Interval: {qt:.1f} ms" if qt else "QT Interval: --",
        f"QTc (Bazett): {qtc:.1f} ms" if qtc else "QTc (Bazett): --"
    ]

    y0 = 30
    for i, line in enumerate(text_lines):
        y = y0 + i * 30
        cv2.putText(annotated, line, (10, y), font, font_scale, color, thickness)

    return annotated
