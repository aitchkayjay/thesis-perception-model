import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
from kitti_labels import labelId_to_trainId
#first trial:
#class KittiDataset(Dataset):
 #   def __init__(self, image_dir, mask_dir, transform=None):
  #      self.image_dir = image_dir
   #     self.mask_dir = mask_dir
    #    self.transform = transform
     #   self.images = sorted([
      #      f for f in os.listdir(image_dir)
       #     if f.endswith(".png")
        #])
 #
  #  def __len__(self):
   #     return len(self.images)
 #
  #  def __getitem__(self, idx):
   #  img_filename = self.images[idx]
    # img_path = os.path.join(self.image_dir, img_filename)
     #mask_path = os.path.join(self.mask_dir, img_filename)
 #
  #   image = Image.open(img_path).convert("RGB")
   #  mask = Image.open(mask_path).convert("L")
 #
  #   if self.transform:
   #     image = self.transform(image)
    #    mask = self.transform(mask)
 #
  #   return image, mask
#second trial:
#class KittiDataset(Dataset):
 #   def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
  #      self.image_dir = image_dir
   #     self.mask_dir = mask_dir
    #    self.transform_img = transform_img
     #   self.transform_mask = transform_mask
      #  self.images = sorted([
       #     f for f in os.listdir(image_dir)
        #    if f.endswith(".png")
        #])
 
  #  def __len__(self):
   #     return len(self.images)
 #
  #  def __getitem__(self, idx):
   #     img_filename = self.images[idx]
    #    img_path = os.path.join(self.image_dir, img_filename)
     #   mask_path = os.path.join(self.mask_dir, img_filename)
#
 #       image = Image.open(img_path).convert("RGB")
  #      mask = Image.open(mask_path)  # mode 'L' is not enforced for class labels
#
 #       if self.transform_img:
  #          image = self.transform_img(image)
   #     if self.transform_mask:
    #        mask = self.transform_mask(mask)
#
 #       mask = torch.from_numpy(np.array(mask)).long()  # semantic class indices
#
 #       return image, mask
#labelId_to_trainId = {
 #   0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
  #  5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
   # 10: 2, 11: 3, 12: 4, 13: 5, 14: 255,
    #15: 255, 16: 255, 17: 6, 18: 7, 19: 8,
   # 20: 9, 21: 10, 22: 11, 23: 12, 24: 13,
   # 25: 14, 26: 15, 27: 16, 28: 17, 29: 18,
   # 30: 255, 31: 255, 32: 255, 33: 255
#}

#trainId_to_label = {
 #   0: ("road",          [128, 64,128]),
  #  1: ("sidewalk",      [244, 35,232]),
   # 2: ("building",      [70, 70, 70]),
    #3: ("wall",          [102,102,156]),
    #4: ("fence",         [190,153,153]),
   # 5: ("pole",          [153,153,153]),
   # 6: ("traffic light", [250,170, 30]),
   # 7: ("traffic sign",  [220,220, 0]),
    #8: ("vegetation",    [107,142, 35]),
    #9: ("terrain",       [152,251,152]),
    #10:("sky",           [70,130,180]),
    #11:("person",        [220, 20, 60]),
    #12:("rider",         [255, 0, 0]),
    #13:("car",           [0, 0,142]),
    #14:("truck",         [0, 0, 70]),
    #15:("bus",           [0, 60,100]),
    #16:("train",         [0, 80,100]),
    #17:("motorcycle",    [0, 0,230]),
    #18:("bicycle",       [119, 11, 32]),
    #255:("void",         [0, 0, 0])
#}
#third trial:
#class KittiDataset(Dataset):
 #   def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
  #      self.image_dir = image_dir
   #     self.mask_dir = mask_dir
    #    self.transform_img = transform_img
     #   self.transform_mask = transform_mask
      #  self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

#    def __len__(self):
 #       return len(self.images)

 #   def __getitem__(self, idx):
  #      img_filename = self.images[idx]
   #     img_path = os.path.join(self.image_dir, img_filename)
    #    mask_path = os.path.join(self.mask_dir, img_filename)
#
 #       image = Image.open(img_path).convert("RGB")
  #      mask = Image.open(mask_path)
#
 #       if self.transform_img:
  #          image = self.transform_img(image)
   #     if self.transform_mask:
    #        mask = self.transform_mask(mask)
#
 #       mask = np.array(mask)
  #      mask = np.vectorize(labelId_to_trainId.get)(mask)
   #     mask = torch.from_numpy(mask).long()
#
 #       return image, mask
NUM_CLASSES = max(labelId_to_trainId.values()) + 1  # z. B. 4

class KittiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_img=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_path = os.path.join(self.mask_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # Label-Mapping absichern
        mask = np.array(mask)
        mask = np.vectorize(lambda x: labelId_to_trainId.get(x, 255))(mask)  # void bleibt 255
        mask = torch.from_numpy(mask).long()

        # Debug-Ausgabe (einmal prüfen, dann wieder auskommentieren)
        #print("Mask min:", mask.min().item(), "max:", mask.max().item(), "unique:", mask.unique())

        return image, mask