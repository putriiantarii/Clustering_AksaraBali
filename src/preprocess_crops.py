
from pathlib import Path
from PIL import Image, ImageOps
import csv

ROOT_DIR = Path(__file__).resolve().parent.parent
CROP_DIR = ROOT_DIR / "data" / "processed" / "character_crops"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 224

metadata_path = CROP_DIR / "metadata.csv"
output_metadata = OUTPUT_DIR / "metadata.csv"

processed_rows = []
failed = 0

with open(metadata_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for row in rows:
    image_path = CROP_DIR / row["crop_file"]

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            # Pertahankan rasio aspek dan tambahkan padding
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

print("=== HASIL PREPROCESSING ===")
print(f"Crop awal       : {len(rows)}")
print(f"Berhasil        : {len(processed_rows)}")
print(f"Gagal           : {failed}")
print(f"Folder output   : {OUTPUT_DIR}")
