import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

# ----- TrainID color and label mapping -----
trainId_to_label = {
    0: ("road",          [128, 64,128]),
    1: ("sidewalk",      [244, 35,232]),
    2: ("building",      [ 70, 70, 70]),
    3: ("wall",          [102,102,156]),
    4: ("fence",         [190,153,153]),
    5: ("pole",          [153,153,153]),
    6: ("traffic light", [250,170, 30]),
    7: ("traffic sign",  [220,220,  0]),
    8: ("vegetation",    [107,142, 35]),
    9: ("terrain",       [152,251,152]),
    10:("sky",           [ 70,130,180]),
    11:("person",        [220, 20, 60]),
    12:("rider",         [255,  0,  0]),
    13:("car",           [  0,  0,142]),
    14:("truck",         [  0,  0, 70]),
    15:("bus",           [  0, 60,100]),
    16:("train",         [  0, 80,100]),
    17:("motorcycle",    [  0,  0,230]),
    18:("bicycle",       [119, 11, 32]),
    255:("void",         [  0,  0,  0])
}

# Label ID to Train ID mapping (KITTI)
labelId_to_trainId = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
    5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
    10: 2, 11: 3, 12: 4, 13: 5, 14: 255,
    15: 255, 16: 255, 17: 6, 18: 7, 19: 8,
    20: 9, 21: 10, 22: 11, 23: 12, 24: 13,
    25: 14, 26: 15, 27: 16, 28: 17, 29: 18,
    30: 255, 31: 255, 32: 255, 33: 255
}

# Fast remapping via lookup table
def remap_label_ids(mask_np):
    lut = np.full(256, 255, dtype=np.uint8)
    for k, v in labelId_to_trainId.items():
        lut[k] = v
    return lut[mask_np]

# Dataset class
class KittiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, img_file)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        mask_np = np.array(mask)
        mask_remapped = remap_label_ids(mask_np)

        return image, mask_remapped  # image: tensor, mask: numpy array

# ----- Visualization -----
def visualize(image_tensor, mask_np, save_path="sample_colored_output.png"):
    image_np = image_tensor.permute(1, 2, 0).numpy()
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    legend_patches = []

    for train_id, (name, color) in trainId_to_label.items():
        if train_id in mask_np:
            color_mask[mask_np == train_id] = color
            patch = mpatches.Patch(color=np.array(color) / 255.0, label=name)
            legend_patches.append(patch)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title("KITTI Colored Semantic Mask")
    plt.axis("off")
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# ----- Main Execution -----
if __name__ == "__main__":
    image_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
    mask_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = KittiDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for img, mask in loader:
        img = img[0]
        mask_np = mask[0].numpy()
        visualize(img, mask_np)
        break