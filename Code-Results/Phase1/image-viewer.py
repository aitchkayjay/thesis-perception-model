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
import numpy as np

# Train ID to RGB (from KITTI dataset setup)
kitti_trainid2label = {
    0:  ("road",           [128, 64,128]),
    1:  ("sidewalk",       [244, 35,232]),
    2:  ("building",       [ 70, 70, 70]),
    3:  ("wall",           [102,102,156]),
    4:  ("fence",          [190,153,153]),
    5:  ("pole",           [153,153,153]),
    6:  ("traffic light",  [250,170, 30]),
    7:  ("traffic sign",   [220,220,  0]),
    8:  ("vegetation",     [107,142, 35]),
    9:  ("terrain",        [152,251,152]),
    10: ("sky",            [ 70,130,180]),
    11: ("person",         [220, 20, 60]),
    12: ("rider",          [255,  0,  0]),
    13: ("car",            [  0,  0,142]),
    14: ("truck",          [  0,  0, 70]),
    15: ("bus",            [  0, 60,100]),
    16: ("train",          [  0, 80,100]),
    17: ("motorcycle",     [  0,  0,230]),
    18: ("bicycle",        [119, 11, 32]),
    255:("void",           [  0,  0,  0])
}

kitti_labels = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle", "void"
]
mask_path = "/mnt/data1/datasets/kitti/kitti_semantics/training/semantic/000124_10.png"
mask = np.array(Image.open(mask_path))
mask = np.where(np.isin(mask, list(kitti_trainid2label.keys())), mask, 255)
# Create an RGB image where each pixel's color is determined by the class index
color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
legend_patches = []

# Fill in the color mask and build legend
for label_id in np.unique(mask):
    if label_id in kitti_trainid2label:
        name, color = kitti_trainid2label[label_id]
        color_mask[mask == label_id] = color
        patch = mpatches.Patch(color=np.array(color) / 255.0, label=name)
        legend_patches.append(patch)

# Plot after color assignment
plt.figure(figsize=(10, 5))
plt.imshow(color_mask)
plt.title("KITTI Colored Semantic Mask")
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.axis("off")
plt.tight_layout()
plt.savefig("segmentation_mask_3.png")