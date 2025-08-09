import numpy as np
from PIL import Image
import os

# === Input and Output Paths ===
mask_path = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks/000009_10.png"  # grayscale mask
output_color_path = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-masks/000009_10_color.png"
os.makedirs(os.path.dirname(output_color_path), exist_ok=True)

# === Load grayscale CARLA mask ===
mask = np.array(Image.open(mask_path))

# === Official CARLA color palette (Cityscapes-style)
carla_palette = {
    0: (0, 0, 0),           # unlabeled/void
    1: (70, 70, 70),        # building
    2: (100, 40, 40),       # fence
    3: (55, 90, 80),        # other
    4: (220, 20, 60),       # pedestrian
    5: (153, 153, 153),     # pole
    6: (157, 234, 50),      # road line
    7: (128, 64, 128),      # road
    8: (244, 35, 232),      # sidewalk
    9: (107, 142, 35),      # vegetation
    10: (0, 0, 142),        # vehicle
    11: (102, 102, 156),    # wall
    12: (220, 220, 0),      # traffic sign
    13: (70, 130, 180),     # sky
    14: (81, 0, 81),        # ground
    15: (150, 100, 100),    # bridge
    16: (230, 150, 140),    # rail track
    17: (180, 165, 180),    # guard rail
    18: (250, 170, 30),     # traffic light
    19: (110, 190, 160),    # static
    20: (170, 120, 50),     # dynamic
    21: (45, 60, 150),      # water
    22: (152, 251, 152),    # terrain
}

# === Colorize mask
def apply_carla_palette(mask_np, palette):
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for class_id in np.unique(mask_np):
        if class_id in palette:
            color_mask[mask_np == class_id] = palette[class_id]
        else:
            print(f"⚠️ Warning: Class ID {class_id} not in palette. Coloring as black.")
    return color_mask

# === Apply palette
unique_ids = np.unique(mask)
print("🔍 Unique class IDs in grayscale mask:", unique_ids)
color_mask = apply_carla_palette(mask, carla_palette)

# === Save colorized image
Image.fromarray(color_mask).save(output_color_path)
print(f"✅ Color mask saved to: {output_color_path}")