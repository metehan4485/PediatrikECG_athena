from PIL import Image, ImageDraw, ImageFont
import numpy as np

def analyze_ekg(image):
    # Görseli gri tona çevir (örnek analiz gibi)
    gray_image = image.convert("L").convert("RGB")

    # Örnek çizim: görselin üstüne "R" işareti koy
    draw = ImageDraw.Draw(gray_image)
    width, height = gray_image.size
    draw.text((width//2 - 20, height//2 - 10), "R", fill="red")

    # Sahte yorum
    yorum = "Sinüs ritmi olabilir, ancak net yorum için algoritma eğitimi gerekiyor."

    return gray_image, yorum
