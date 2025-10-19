
import torch

from torch.utils.data import Dataset
from torch.utils.data import random_split
from PIL import Image 

import matplotlib.pyplot as plt

import numpy as np

import os

from tqdm import tqdm # for the progress bar

from hyper import *
from augmentation import transofrm_train

def fetch_data(path) -> list:
  """Return files as their paths+filename in an array"""

  assert (os.path.exists(path) == True),  "Failure during data fetching"  
      
  result = []
  for file in tqdm(os.listdir(path), desc=f"Loading files from {path} ",unit="File", leave=True):
    fpath = os.path.join(path,file)
    result.append(fpath)
  
  return result

class DeepCrackDataset(Dataset):
  def __init__(self, img_dir, mask_dir, transform=None):
    
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.transform = transform

    # sort values so the file names corespoding to each other are loaded in order
    self.images = sorted([os.path.join(img_dir, file) for file in os.listdir(img_dir)] )
    self.masks = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)]) 

  def __len__(self):
    """Return the length of images(and masks) dataset"""
    return len(self.images)

  def __getitem__(self, index):
    """Returns the image and mask as Tensor data as per requested index --idx,
      """
    np_image = np.array(Image.open(self.images[index]))
    np_mask = np.array(Image.open(self.masks[index])) 

   
    if len(np_mask.shape) == 3:
      np_mask = np_mask[:,:,0]

    np_mask = (np_mask > 127).astype(np.uint8)
    
    if self.transform: # if using transforms
      t = self.transform(image=np_image, mask=np_mask)
      np_image = t["image"]
      np_mask = t["mask"]

    # conversion from numpy array convention to tensor via permute, then normalizing to [0,1] range, same for mask, only using binary data
    tensor_image = torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0
    tensor_mask = torch.from_numpy(np_mask).unsqueeze(0).float() 

    return tensor_image,tensor_mask
  
def get_dataset(img_path = IMAGE_PATH, mask_path = MASK_PATH ):
  
  dataset = DeepCrackDataset(img_path, mask_path, transform=transofrm_train)
  return dataset

def split_dataset(dataset: DeepCrackDataset, train_factor, test_factor, val_factor )->list:
  """Split exising dataset given percentages as [0,1] floats, return list of  """
  train_set_len, test_set_len, val_set_len = int(dataset.__len__() * train_factor), int(dataset.__len__() * test_factor) , int(dataset.__len__() * val_factor)
  train_set, test_set ,val_set = random_split(dataset, [train_set_len, test_set_len, val_set_len])
  
  return [train_set, test_set, val_set]


def show_dataset(data_loader, samples=4):
    counter = 0
    # Wczytaj jeden batch
    for images, masks in data_loader:
        
        span = np.ceil(np.sqrt(samples))
        
        fig, axes = plt.subplots(samples, 2, figsize=(8, 12))

        for i in range( samples ):
            
            img = images[i].permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis('off')

            # # Maska
            mask = masks[i, 0].numpy()
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f"Mask {i+1}")
            axes[i, 1].axis('off')
            
        plt.tight_layout()
        plt.show()
        counter+=1
        

