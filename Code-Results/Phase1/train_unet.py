import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from unet import UNet
from kitti_dataset import KittiDataset

# ---- Konfiguration ----
image_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
mask_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
batch_size = 4
num_classes = 34  # KITTI Label IDs von 0–33 (255 wird ignoriert)
lr = 1e-4
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Transformationen ----
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Für Masken: nur Resize, ohne ToTensor
def transform_mask_fn(mask):
    return mask.resize((256, 256), resample=Image.NEAREST)

# ---- Dataset & DataLoader ----
dataset = KittiDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform_img=transform_img,
    transform_mask=transform_mask_fn
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---- Modell, Loss, Optimizer ----
model = UNet(n_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignoriere "void"-Klasse
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---- Training ----
best_loss = float("inf")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        masks = masks.to(device)  # [B, H, W]

        optimizer.zero_grad()
        outputs = model(images)  # [B, C, H, W]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1} – Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Bestes Modell gespeichert: best_model.pth")

print("🎉 Training abgeschlossen.")