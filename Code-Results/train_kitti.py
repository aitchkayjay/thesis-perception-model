import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import logging
from pathlib import Path

from unet.unet_model import UNet
from kitti_dataset import KittiDataset
from kitti_labels import NUM_CLASSES

# Pfade zu deinen Daten
image_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
mask_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform_img = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
transform_mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)

# Dataset & Loader
dataset = KittiDataset(image_dir, mask_dir, transform_img=transform_img, transform_mask=transform_mask)
train_len = int(len(dataset) * 0.9)
val_len = len(dataset) - train_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)

# Modell
model = UNet(n_channels=3, n_classes=NUM_CLASSES)
model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 20
best_loss = float('inf')
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    logging.info(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), f"/home/husjaa/Pytorch-UNet/checkpoints/unet_epoch{epoch+1}.pth")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "/home/husjaa/Pytorch-UNet/checkpoints/best_model.pth")
        print(f"✅ Best model saved at epoch {epoch+1} with loss {epoch_loss:.4f}")