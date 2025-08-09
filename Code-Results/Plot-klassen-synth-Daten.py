from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Bildpfad
img_path = "/mnt/data1/synthetic_data/synthetic_semseg/labels/t00_000315.png"
output_dir = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-masks/saved_masks"
os.makedirs(output_dir, exist_ok=True)

# Bild laden
mask = Image.open(img_path)
mask_np = np.array(mask)

# Anzeigen
plt.figure(figsize=(6, 6))
plt.imshow(mask_np, cmap="nipy_spectral")  # cmap für semantische Labels
plt.title("Semantische Maske")
plt.axis("off")

# Bild speichern
save_path = os.path.join(output_dir, "t00_000315_mask_visualized.png")
plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
print(f"✅ Maske gespeichert unter: {save_path}")

# Maskenwerte analysieren
unique_classes = np.unique(mask_np)
print("🎯 Einzigartige Klassen-IDs in der Maske:", unique_classes)
# Optional: Maske als Rohbild (z. B. für spätere Verarbeitung) speichern
raw_save_path = os.path.join(output_dir, "t00_000315_mask_raw.png")
Image.fromarray(mask_np).save(raw_save_path)
print(f"📦 Rohe Masken-Datei gespeichert unter: {raw_save_path}")