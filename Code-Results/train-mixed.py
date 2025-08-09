import os
import math
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from unet import UNet
from utils.dice_score import dice_loss

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

NUM_CLASSES = 24                 # CARLA IDs: 0..23 (0 = void)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-4
VAL_SPLIT = 0.1                  # 10% of the concatenated dataset for validation
BALANCE_DOMAINS = True           # use WeightedRandomSampler to balance KITTI vs CARLA

# --- Paths (adjust these!) ---
# KITTI (images + your remapped CARLA-ID masks)
KITTI_IMAGE_DIR = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
KITTI_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Kitti-2-Carla-masks"

# CARLA (images + your converted ID masks)
CARLA_IMAGE_DIR = "/mnt/data1/synthetic_data/synthetic_semseg/images"
CARLA_MASK_DIR  = "/mnt/data1/Hussein-thesis-repo/Code-Results/Carla-ID-masks"

CHECKPOINT_DIR  = "/mnt/data1/Hussein-thesis-repo/Checkpointss/Checkpoint-mixed"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "unet_kitti_plus_carla_best.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# Transforms
# =========================
transform_img = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_mask = transforms.Compose([
    transforms.Resize((256, 512), interpolation=transforms.InterpolationMode.NEAREST)
])

# =========================
# Dataset
# =========================
class FolderSegDataset(Dataset):
    """
    Generic dataset: matches image filenames in image_dir to mask_dir 1:1.
    Masks are ID images (uint8) with class IDs 0..23 (void=0).
    """
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None, domain_name=""):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.domain_name = domain_name  # "KITTI" or "CARLA" (for balancing / debug)

        # Only keep files that exist in both folders
        imgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.filenames = [f for f in imgs if os.path.exists(os.path.join(mask_dir, os.path.splitext(f)[0] + ".png"))]
        # prefer PNG masks
        # If your masks use same extension as images, you can just use f as-is

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path  = os.path.join(self.image_dir, fname)
        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)  # grayscale ID mask

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        return image, mask, self.domain_name

# =========================
# Build datasets
# =========================
kitti_ds = FolderSegDataset(
    KITTI_IMAGE_DIR, KITTI_MASK_DIR,
    transform_img=transform_img, transform_mask=transform_mask, domain_name="KITTI"
)

carla_ds = FolderSegDataset(
    CARLA_IMAGE_DIR, CARLA_MASK_DIR,
    transform_img=transform_img, transform_mask=transform_mask, domain_name="CARLA"
)

full_ds = ConcatDataset([kitti_ds, carla_ds])

# Train/Val split (on the concatenated dataset indices)
val_len = max(1, int(len(full_ds) * VAL_SPLIT))
train_len = len(full_ds) - val_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                generator=torch.Generator().manual_seed(SEED))

def make_loader(dataset, shuffle, batch_size=BATCH_SIZE, balance_domains=False):
    if not balance_domains or not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    # Weighted sampler to balance KITTI vs CARLA in the TRAIN split
    # Build domain labels array for the subset
    domain_weights = []
    # We need to resolve per-sample which underlying dataset it came from
    # ConcatDataset stores datasets list and sample is an index into one of them
    # For items in Subset(dataset, indices), we map index back to (which ds?, idx_in_ds)
    for idx in dataset.indices:  # dataset is a Subset
        # Find which original dataset this index belongs to
        if idx < len(kitti_ds):
            # KITTI
            domain_weights.append(0)  # mark as KITTI
        else:
            domain_weights.append(1)  # CARLA

    domain_weights = np.array(domain_weights)
    # inverse-frequency weights
    n0 = (domain_weights == 0).sum()
    n1 = (domain_weights == 1).sum()
    w0 = 0 if n0 == 0 else 0.5 / n0
    w1 = 0 if n1 == 0 else 0.5 / n1
    sample_weights = np.where(domain_weights == 0, w0, w1).astype(np.float32)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)

train_loader = make_loader(train_ds, shuffle=True,  balance_domains=BALANCE_DOMAINS)
val_loader   = make_loader(val_ds,   shuffle=False, balance_domains=False)

print(f"KITTI samples: {len(kitti_ds)} | CARLA samples: {len(carla_ds)}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# =========================
# Model / Loss / Optim
# =========================
model = UNet(n_channels=3, n_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # void=0 ignored

def step_batch(images, masks):
    preds = model(images)
    # CE expects [N, C, H, W] logits and [N, H, W] long
    loss = ce_loss(preds, masks) + dice_loss(preds, masks, multiclass=True)
    return loss

# =========================
# Train
# =========================
best_val = math.inf
for epoch in range(1, EPOCHS + 1):
    # --- train ---
    model.train()
    running = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]")
    for images, masks, _domain in pbar:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = step_batch(images, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    train_loss = running / max(1, len(train_loader))

    # --- val ---
    model.eval()
    running = 0.0
    with torch.no_grad():
        for images, masks, _domain in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            loss = step_batch(images, masks)
            running += loss.item()
    val_loss = running / max(1, len(val_loader))

    print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"✅ Saved best model → {CHECKPOINT_PATH}")