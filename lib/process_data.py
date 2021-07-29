from pathlib import Path
from PIL import Image
import numpy as np
import os



def process(path, mask_path=None):
    data = []
    if mask_path:
        mask_path = str(Path(mask_path))
    images = os.listdir(path)

    for k in images:
      ind = k.split(".")[0]

      item = {}

      img = np.array(Image.open(f"{path}/{ind}.jpg").convert("RGB"))
      item['img'] = img
      
      if mask_path:
        mask = np.array(Image.open(f"{mask_path}/{ind}.png"), dtype=np.float32)[:,:,None]
        item['mask'] = mask.clip(max=1)
      
      data.append(item)
    
    return data


class Dataset_With_Transforms():
    def __init__(self, data, transforms, mask_present=True):
        self.datas = data
        self.transforms = transforms
        self.mask_present = mask_present

    def __getitem__(self, index):
        data = self.datas[index]

        if not self.mask_present:
          return self.transforms(image=data['img'])['image']

        img, mask = data['img'], data['mask']
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        return img, mask
        
    def __len__(self):
        return len(self.datas)



def get_data(path, mask_path, augmentation):
    data = process(path, mask_path)
    dataset = Dataset_With_Transforms(data, augmentation, mask_path is not None)
    X_train = [dataset[i] for i in range(len(dataset))]
    return X_train

