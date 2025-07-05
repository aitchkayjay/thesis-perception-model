import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from unet import UNet
from kitti_dataset import KittiDataset  # deine eigene Klasse
from kitti_labels import labelId_to_trainId, trainId_to_label  # Stelle sicher, dass sie importierbar sind

# ------------------------------
# Konfiguration   j
# ------------------------------
MODEL_PATH = "/home/husjaa/Pytorch-UNet/checkpoints/best_model.pth"
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)
NUM_CLASSES = 19  # oder len([k for k in trainId_to_label.keys() if k != 255])

# ------------------------------
# Transformations
# ------------------------------
transform_img = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

def transform_mask_fn(mask):
    return mask.resize(IMAGE_SIZE, resample=Image.NEAREST)

# ------------------------------
# Modell laden
# ------------------------------
model = UNet(n_channels=3, n_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ------------------------------
# Dataset & Loader
# ------------------------------
dataset = KittiDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask_fn
)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ------------------------------
# Nimm ein Sample
# ------------------------------
image_tensor, mask_tensor = next(iter(loader))
image_tensor = image_tensor.to(DEVICE)
with torch.no_grad():
    output = model(image_tensor)
prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
gt_mask = mask_tensor.squeeze().cpu().numpy()

# ------------------------------
# Colormap vorbereiten
# ------------------------------
def decode_segmentation(mask):
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for train_id, (name, color) in trainId_to_label.items():
        color_mask[mask == train_id] = color
    return color_mask

color_gt = decode_segmentation(gt_mask)
color_pred = decode_segmentation(prediction)

# ------------------------------
# Legende vorbereiten
# ------------------------------
legend_patches = [
    mpatches.Patch(color=np.array(color)/255.0, label=name)
    for train_id, (name, color) in trainId_to_label.items()
    if train_id != 255
]

# ------------------------------
# Anzeige
# ------------------------------
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_tensor.squeeze().permute(1, 2, 0).cpu())
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(color_gt)
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(color_pred)
plt.title("Model Prediction")
plt.axis("off")

plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("kitti_evaluation_result.png")
plt.show()
