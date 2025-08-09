import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Farbkodierung der CARLA-Masken (RGB → ID) ===
carla_palette = {
    (0, 0, 0): 0,          # Unlabeled / Void
    (70, 70, 70): 1,       # Building
    (190, 153, 153): 2,    # Fence
    (250, 170, 160): 3,    # Other
    (220, 20, 60): 4,      # Pedestrian
    (153, 153, 153): 5,    # Pole
    (157, 234, 50): 6,     # Road Line
    (128, 64, 128): 7,     # Road
    (244, 35, 232): 8,     # Sidewalk
    (107, 142, 35): 9,     # Vegetation
    (0, 0, 142): 10,       # Car
    (102, 102, 156): 11,   # Wall
    (220, 220, 0): 12,     # Traffic Sign
    (70, 130, 180): 13,    # Sky
    (81, 0, 81): 14,       # Ground
    (150, 100, 100): 15,   # Bridge
    (230, 150, 140): 16,   # Rail Track
    (180, 165, 180): 17,   # Guard Rail
    (250, 170, 30): 18,    # Traffic Light
    (0, 0, 110): 19,       # Trailer
    (0, 60, 100): 20,      # Bus
    (0, 0, 90): 21,        # Caravan
    (0, 0, 230): 22,       # Motorcycle
    (119, 11, 32): 23,     # Bicycle
}

# === Ordnerpfade ===
input_mask_dir = "/mnt/data1/synthetic_data/synthetic_semseg/labels"
output_mask_dir = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"
os.makedirs(output_mask_dir, exist_ok=True)

# === Hilfsfunktion: RGB zu Class-ID ===
def rgb_to_class_id(rgb_mask, palette):
    h, w, _ = rgb_mask.shape
    id_mask = np.zeros((h, w), dtype=np.uint8)
    for color, class_id in palette.items():
        mask = np.all(rgb_mask == color, axis=-1)
        id_mask[mask] = class_id
    return id_mask

# === Umwandlung aller RGB-Masken ===
for fname in tqdm(sorted(os.listdir(input_mask_dir))):
    if not fname.endswith(".png"):
        continue
    rgb_path = os.path.join(input_mask_dir, fname)
    rgb_mask = np.array(Image.open(rgb_path).convert("RGB"))  # Sicherstellen: 3-Kanal-Bild
    id_mask = rgb_to_class_id(rgb_mask, carla_palette)

    out_path = os.path.join(output_mask_dir, fname)
    Image.fromarray(id_mask).save(out_path)

print("✅ Alle RGB-Masken erfolgreich in ID-Masken umgewandelt.")
print("📂 Gespeichert unter:", output_mask_dir)