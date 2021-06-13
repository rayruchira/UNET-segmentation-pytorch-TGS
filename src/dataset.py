# packages
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader

# for splitting it externally
# needs to be merged with dataset class

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

# dataset = ImageFolder('C:\Datasets\lcms-dataset', transform=Compose([Resize((224,224)),ToTensor()]))
# print(len(dataset))
# datasets = train_val_dataset(dataset)
# print(len(datasets['train']))
# print(len(datasets['val']))
# # The original dataset is available in the Subset class
# print(datasets['train'].dataset)

# dataloaders = {x:DataLoader(datasets[x],32, shuffle=True, num_workers=4) for x in ['train','val']}
# x,y = next(iter(dataloaders['train']))
# print(x.shape, y.shape)


class TGSdataset(Dataset):
  def __init__(self, img, mask, transform=None):
    #basic logging of directories img = og image directory, mask= mask of images directory
    self.imgP=img
    self.maskP=mask
    self.transform=transform

    #total no. of images
    self.images=os.listdir(img)

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self,index):
    imgpath=os.path.join(self.imgP, self.images[index])
    maskpath=os.path.join(self.maskP, self.images[index]) #name of mask file same as normal file or would have to edit the images[index]part
    image=np.arry(Image.open(imgpath).convert("L"), dtype=np.float32) #might not be needed, could already be in grayscale( could also be RGB)
    mask=np.arry(Image.open(imgpath).convert("L"), dtype=np.float32) #might not be needed, could already be in grayscale
    mask[mask== 255.0] =1.0 #change 255 entries to 1, as we will use sigmid for last activation, 1 is white

    if self.transform is not None:
      augmentations= self.transform(image=image, mask=mask)
      image =augmentations["image"]
      mask=augmentations["mask"]

    return image,mask
    
  def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

