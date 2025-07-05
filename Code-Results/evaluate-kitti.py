import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from unet import UNet
from kitti_dataset import KittiDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Dataset-Pfade
image_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
mask_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"

# Transform für Evaluation
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset & Loader
dataset = KittiDataset(image_dir=image_dir, mask_dir=mask_dir, transform_img=img_transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Modell laden
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=19).to(device)  # Passe n_classes ggf. an!
model.load_state_dict(torch.load("/home/husjaa/Pytorch-UNet/checkpoints/best_model.pth", map_location=device))
model.eval()

# Eine Vorhersage visualisieren
with torch.no_grad():
    for images, masks in loader:
        images = images.to(device)
        output = model(images)
        pred = output.argmax(1).squeeze().cpu().numpy()
        img_np = images.squeeze().permute(1, 2, 0).cpu().numpy()

        # Plot original + prediction
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Input Image")

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap="nipy_spectral")
        plt.title("Predicted Segmentation")

        plt.tight_layout()
        plt.savefig("output_prediction.png")
        break  # Nur ein Bild anzeigen
