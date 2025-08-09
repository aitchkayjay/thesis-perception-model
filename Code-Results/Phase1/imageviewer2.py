import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# KITTI class colors (19 classes from the semantic dataset)
kitti_colors = np.array([
    [  0,   0,   0],    # 0: unlabeled
    [128, 64,128],      # 1: road
    [244, 35,232],      # 2: sidewalk
    [ 70, 70, 70],      # 3: building
    [102,102,156],      # 4: wall
    [190,153,153],      # 5: fence
    [153,153,153],      # 6: pole
    [250,170, 30],      # 7: traffic light
    [220,220,  0],      # 8: traffic sign
    [107,142, 35],      # 9: vegetation
    [152,251,152],      # 10: terrain
    [ 70,130,180],      # 11: sky
    [220, 20, 60],      # 12: person
    [255,  0,  0],      # 13: rider
    [  0,  0,142],      # 14: car
    [  0,  0, 70],      # 15: truck
    [  0, 60,100],      # 16: bus
    [  0, 80,100],      # 17: train
    [  0,  0,230],      # 18: motorcycle
    [119, 11, 32],      # 19: bicycle
], dtype=np.uint8)

kitti_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle", "void"
]
mask_path = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic/000124_10.png"
mask = np.array(Image.open(mask_path))
mask[mask >= len(kitti_colors)] = len(kitti_colors) - 1
# Create an RGB image where each pixel's color is determined by the class index
color_mask = kitti_colors[mask]

# Display the colored mask
plt.figure(figsize=(10, 5))
plt.imshow(color_mask)
plt.title("KITTI Colored Semantic Mask")
legend_elements = [
    mpatches.Patch(color=np.array(c)/255, label=l) for c, l in zip(kitti_colors, kitti_labels)
]

# Show legend
plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.axis("off")
# Load original image
img_path = "/mnt/data1/datasets/kitti/kitti_semantics/training/image_2/000124_10.png"
image = np.array(Image.open(img_path).convert("RGB"))

# Blend original image and color mask (alpha=0.5 means 50% transparency)
blended = (0.5 * image + 0.5 * color_mask).astype(np.uint8)

# Show side-by-side
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(color_mask)
plt.title("Colored Segmentation")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blended)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.savefig("segmentation_mask_1.png")  # Saves image to file