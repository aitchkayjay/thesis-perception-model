import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

# === Konfiguration ===
NUM_CLASSES = 24  # CARLA hat IDs von 0 bis 23
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIR = "/mnt/data1/synthetic_data/synthetic_semseg/images"
MASK_DIR = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"
CHECKPOINT_PATH = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mit-synthData/unet_carla_best.pth"
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# === Transforms ===
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# === Dataset ===
class CarlaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(os.listdir(image_dir))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)  # already grayscale with class IDs

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

# === DataLoader ===
dataset = CarlaDataset(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    transform_img=transform_img,
    transform_mask=transform_mask
)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# === Modell, Optimizer, Loss ===
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 = void

# === Training Loop ===
epochs = 50
best_loss = float('inf')

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Modell gespeichert unter: {CHECKPOINT_PATH}")