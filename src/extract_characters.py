
from pathlib import Path
import sys
import csv
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import RAW_IMAGE_DIR, RAW_LABEL_DIR, PROCESSED_DIR

IMAGE_DIR = Path(RAW_IMAGE_DIR)
LABEL_DIR = Path(RAW_LABEL_DIR)
OUTPUT_DIR = Path(PROCESSED_DIR) / "character_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_characters():
    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    metadata = []
    total_boxes = 0
    valid_crops = 0
    skipped = 0

    for image_path in image_files:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Label tidak ditemukan: {image_path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Gagal membaca gambar: {image_path.name}")
            skipped += 1
            continue

        image_h, image_w = image.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            parts = line.strip().split()

            if not parts:
                continue

            if len(parts) != 5:
                print(
                    f"Format anotasi tidak valid: "
                    f"{label_path.name}, baris {line_idx + 1}"
                )
                skipped += 1
                continue

            try:
                class_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
            except ValueError:
                skipped += 1
                continue

            total_boxes += 1

            # Validasi koordinat YOLO
            if (
                not all(map(lambda v: 0 <= v <= 1, [xc, yc, bw, bh]))
                or bw <= 0 or bh <= 0
            ):
                print(
                    f"Koordinat tidak valid: "
                    f"{image_path.name}, baris {line_idx + 1}"
                )
                skipped += 1
                continue

            # Konversi koordinat relatif ke piksel
            x1 = round((xc - bw / 2) * image_w)
            y1 = round((yc - bh / 2) * image_h)
            x2 = round((xc + bw / 2) * image_w)
            y2 = round((yc + bh / 2) * image_h)

            # Pastikan koordinat berada dalam batas gambar
            x1 = max(0, min(x1, image_w))
            y1 = max(0, min(y1, image_h))
            x2 = max(0, min(x2, image_w))
            y2 = max(0, min(y2, image_h))

            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            crop = image[y1:y2, x1:x2]

            # Hindari crop yang terlalu kecil
            crop_h, crop_w = crop.shape[:2]
            if crop_w < 4 or crop_h < 4:
                skipped += 1
                continue

            crop_name = (
                f"{image_path.stem}_box{line_idx:04d}.jpg"
            )
            crop_path = OUTPUT_DIR / crop_name

            if not cv2.imwrite(str(crop_path), crop):
                print(f"Gagal menyimpan crop: {crop_name}")
                skipped += 1
                continue

            metadata.append({
                "crop_file": crop_name,
                "source_image": image_path.name,
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

            valid_crops += 1

    csv_path = OUTPUT_DIR / "metadata.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "crop_file", "source_image", "class_id",
            "x1", "y1", "x2", "y2"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print("\n=== HASIL EKSTRAKSI ===")
    print(f"Jumlah gambar sumber : {len(image_files)}")
    print(f"Total anotasi dibaca : {total_boxes}")
    print(f"Crop berhasil        : {valid_crops}")
    print(f"Data dilewati        : {skipped}")
    print(f"Folder hasil         : {OUTPUT_DIR}")
    print(f"Metadata             : {csv_path}")


if __name__ == "__main__":
    extract_characters()
