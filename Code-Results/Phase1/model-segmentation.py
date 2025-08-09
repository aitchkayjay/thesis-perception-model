import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from unet import UNet
from kitti_dataset import labelId_to_trainId, trainId_to_label  # falls vorhanden

# ---------- Pfade ----------
model_path = "best_model.pth"
image_path = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2/000124_10.png"

# ---------- Konfiguration ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 34

# ---------- Bildvorbereitung ----------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Lade und transformiere das Bild
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ---------- Modell vorbereiten ----------
model = UNet(n_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- Vorhersage ----------
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H,W]

# ---------- Farb-Mapping ----------
color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
legend_patches = []

for train_id, (label, color) in trainId_to_label.items():
    if train_id in pred:
        color_mask[pred == train_id] = color
        patch = mpatches.Patch(color=np.array(color)/255.0, label=label)
        legend_patches.append(patch)

# ---------- Anzeigen ----------
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image.resize((256, 256)))
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(color_mask)
plt.title("Vorhergesagte Maske")

plt.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig("predicted_mask.png")
plt.show()