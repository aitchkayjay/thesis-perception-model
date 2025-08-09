import unet
from unet import UNet
import torch
from kitti_dataset import KittiDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
 
# Dataset paths
image_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2"
mask_dir = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic"
 
# Define the dataset
dataset = KittiDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    transform=transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
)
 
# Wrap in DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True)
 
for img, mask in loader:
    plt.figure(figsize=(10, 5))
 
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].permute(1, 2, 0))  # CHW -> HWC
    plt.title("Input Image")
 
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze().numpy(), cmap="gray")
    plt.title("Segmentation Mask")
 
    plt.savefig("sample_output.png")  # <-- saves to file
    break