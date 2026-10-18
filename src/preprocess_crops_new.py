from pathlib import Path
from PIL import Image, ImageOps
import csv
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
CROP_DIR = ROOT_DIR / "data" / "processed" / "character_crops"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "preprocessed_enhanced" 
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 224

metadata_path = CROP_DIR / "metadata.csv"
output_metadata = OUTPUT_DIR / "metadata.csv"

processed_rows = []
failed = 0

def enhance_lontar_image(image_path):
    # Baca gambar format Grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca")

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 5
    )

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    sizes = stats[1:, -1]
    
    clean_mask = np.zeros_like(thresh)
    min_area_threshold = 15 
    
    for i in range(0, nb_components - 1):
        if sizes[i] >= min_area_threshold:
            clean_mask[output == i + 1] = 255

    # Penghalusan Tepi (Edge Smoothing)
    smoothed = cv2.medianBlur(clean_mask, 3)

    # Balikkan warna ke awal (Teks = Hitam, Background = Putih)
    result = cv2.bitwise_not(smoothed)

    # Konversi ke RGB untuk ResNet
    img_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    
    return Image.fromarray(img_rgb)

with open(metadata_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    image_path = CROP_DIR / row["crop_file"]

    try:
        img = enhance_lontar_image(image_path)

        img = ImageOps.pad(
            img,
            (SIZE, SIZE),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5)
        )

        output_path = OUTPUT_DIR / row["crop_file"]
        img.save(output_path)

        processed_rows.append(row)

    except Exception as e:
        print(f"Gagal memproses {image_path.name}: {e}")
        failed += 1

with open(output_metadata, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(processed_rows)

print("=== HASIL PREPROCESSING ENHANCED ===")
print(f"Crop awal       : {len(rows)}")
print(f"Berhasil        : {len(processed_rows)}")
print(f"Gagal           : {failed}")
print(f"Folder output   : {OUTPUT_DIR}")