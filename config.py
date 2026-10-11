import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "DeepLontar")
RAW_LABEL_DIR = os.path.join(BASE_DIR, "data", "raw", "DeepLontar_Labels")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")