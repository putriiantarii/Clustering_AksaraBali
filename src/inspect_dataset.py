import os
import sys
import cv2
import matplotlib.pyplot as plt

# Menambahkan root directory ke path agar bisa membaca config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_IMAGE_DIR, RAW_LABEL_DIR

def check_dataset_health(raw_image_dir, raw_label_dir):
    """
    Fungsi untuk melakukan inspeksi kesehatan awal dataset DeepLontar.
    """
    if not os.path.exists(raw_image_dir) or not os.path.exists(raw_label_dir):
        print("⚠️ Folder dataset di 'data/raw/' belum lengkap atau belum diisi!")
        print(f"Pastikan folder 'DeepLontar' dan 'DeepLontar_Labels' ada di dalam folder 'data/raw/'.")
        return

    images = [f for f in os.listdir(raw_image_dir) if f.endswith(".jpg")]
    labels = [f for f in os.listdir(raw_label_dir) if f.endswith(".txt")]
    
    print("==========================================")
    print("       LAPORAN INSPEKSI DATASET          ")
    print("==========================================")
    print(f" Total file gambar (.jpg): {len(images)}")
    print(f" Total file label (.txt) : {len(labels)}")
    
    if labels:
        sample_label = labels[0]
        sample_path = os.path.join(raw_label_dir, sample_label)
        print(f"\n📄 Contoh isi file label dari ({sample_label}):")
        with open(sample_path, "r") as f:
            lines = f.readlines()
            for i in range(min(3, len(lines))):
                print(f"  Baris {i+1}: {lines[i].strip()}")
        print(f"  Total baris/karakter teranotasi: {len(lines)}")
    print("==========================================")


def visualize_sample(image_name="100a.jpg"):
    """
    Fungsi untuk memuat anotasi bounding box bawaan dataset dan menampilkannya secara visual.
    """
    img_path = os.path.join(RAW_IMAGE_DIR, image_name)
    txt_path = os.path.join(RAW_LABEL_DIR, image_name.replace(".jpg", ".txt"))
    
    if not os.path.exists(img_path) or not os.path.exists(txt_path):
        print(f"⚠️ File sampel {image_name} tidak ditemukan di folder raw!")
        return

    img = cv2.imread(img_path)
    if img is None:
        print("Gagal membaca gambar.")
        return
        
    h_img, w_img, _ = img.shape

    with open(txt_path, "r") as f:
        lines = f.readlines()

    # Membaca koordinat dari file .txt bawaan dataset (Format YOLO)
    for line in lines:
        data = line.strip().split()
        if len(data) >= 5:
            x_center, y_center, w, h = map(float, data[1:5])
            
            # Konversi koordinat YOLO ke piksel absolut gambar
            xmin = int((x_center - w / 2) * w_img)
            ymin = int((y_center - h / 2) * h_img)
            xmax = int((x_center + w / 2) * w_img)
            ymax = int((y_center + h / 2) * h_img)

            # Menggambar kotak hijau tipis di sekeliling karakter aksara
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)

    # Menampilkan hasil visualisasi
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Visualisasi Anotasi Dataset: {image_name} ({len(lines)} Karakter)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Menjalankan pengecekan dataset DeepLontar...")
    check_dataset_health(RAW_IMAGE_DIR, RAW_LABEL_DIR)
    
    # Menampilkan visualisasi kotak anotasi
    print("\nMembuka jendela visualisasi bounding box...")
    visualize_sample("100a.jpg")