import numpy as np
import torch
from torch.utils.data import Dataset
from mypath import Path
from tqdm import trange
import os
from pycocotools import mask
from torchvision import transforms
from dataloaders import custom_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FactorySegmentation():
    NUM_CLASSES = 5
    
    def __init__(self, args, base_dir=Path.db_root_dir('factory'),split='train'):
        self.label_path = glob(base_dir+"label_img/*")
        img_path = [p.split("/")[-1].split(".")[0] for p in self.label_path]
        self.img_path = [base_dir+"data_img/"+num+".png" for num in img_path]
        self.split = split
        
    def __getitem__(self, index):
        _img = np.array(Image.open(self.img_path[index]).convert('RGB'), dtype=np.float32)
        _target = np.array(Image.open(self.label_path[index]).convert('RGB'), dtype=np.float32)
        _target = self._gen_seg_mask(_target)
        sample = {'image': _img, 'label': _target}
        
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
    
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)
    
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()
        ])
        return composed_transforms(sample)
    
    def _gen_seg_mask(self, target):
        target /= 255
        mask = target[:,:,0] + target[:,:,1]*2 + target[:,:,2]*3

        mask = np.where(mask==1, 1, mask)
        mask = np.where(mask==2, 2, mask)
        mask = np.where(mask==3, 3, mask)
        mask = np.where(mask==6, 4, mask)
        return mask
    
    def __len__(self):
        return len(self.img_path)


if __name__ == "__main__":
    from dataloaders import custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    factory_val = FactorySegmentation(args, split='val')

    dataloader = DataLoader(factory_val, batch_size=4, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='factory')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)