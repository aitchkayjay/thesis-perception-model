import numpy as np
from PIL import Image

# === Path to your segmentation image ===
image_path = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic/000009_10.png"  # grayscale mask

# === Load image and convert to numpy array ===
mask = np.array(Image.open(image_path))

# === Get and print unique class IDs ===
unique_ids = np.unique(mask)
print("🔍 Unique KITTI class IDs in image:", unique_ids)