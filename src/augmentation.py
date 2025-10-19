import albumentations as A
import numpy as np


transofrm_train = A.Compose([
    A.RandomResizedCrop(size=(256,256),scale=(0.5, 1.0)),
    A.HorizontalFlip(p=0.5),  
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], seed=np.random.randint(low=1, high=1000))