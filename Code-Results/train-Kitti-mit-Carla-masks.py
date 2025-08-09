import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from KittiCarlaDataset import KittiCarlaDataset
from unet import UNet
from utils.dice_score import dice_loss

# === Constants ===
NUM_CLASSES = 24  # 0–23 from remapped CARLA labels
IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
MASK_DIR = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"  # <- YOUR REMAPPED MASKS
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-Carla-Masks/unet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# === Dataset & DataLoader ===
dataset = KittiCarlaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# === Model, Optimizer, Loss ===
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 = void class in CARLA

# === Training Loop ===
best_loss = float('inf')
epochs = 50
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()

        preds = model(images)
        loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Best model saved at {CHECKPOINT_PATH}")