
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import math

CROP_DIR = Path("data/processed/character_crops")
crop_files = sorted(CROP_DIR.glob("*.jpg"))[:25]

if not crop_files:
    print("Belum ada hasil crop. Jalankan script ekstraksi dahulu.")
else:
    cols = 5
    rows = math.ceil(len(crop_files) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, path in zip(axes, crop_files):
        img = cv2.imread(str(path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(path.name, fontsize=7)
        ax.axis("off")

    for ax in axes[len(crop_files):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
