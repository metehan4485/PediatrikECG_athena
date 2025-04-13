def get_scale_from_points(x1, x2):
    """
    İki dikey çizgi arası piksel mesafesinden zaman ölçeğini hesaplar.

    Parametreler:
    x1, x2: int – iki kalibrasyon çizgisinin x koordinatları

    Dönüş:
    time_scale: float – ms per pixel
    """
    pixels_per_5mm = abs(x2 - x1)
    mm_per_ms = 0.2  # 25 mm/s hızda 1 mm = 0.04 s = 40 ms ⇒ 1 ms = 0.025 mm ⇒ 1 mm = 0.2 ms
    time_scale = (5 / pixels_per_5mm) / mm_per_ms * 1000  # ms per pixel
    return time_scale
