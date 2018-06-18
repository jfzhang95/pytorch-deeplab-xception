from __future__ import print_function, division
import json
import os

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
from mypath import Path


class SBDSegmentation(data.Dataset):

    def __init__(self,
                 base_dir=Path.db_root_dir('sbd'),
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        self._image_dir = os.path.join(self._dataset_dir, 'img')
        self._cat_dir = os.path.join(self._dataset_dir, 'cls')


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform



        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ= os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))



        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):

        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'gt': _target}


        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        _target = np.array(scipy.io.loadmat(self.categories[index])["GTcls"][0]['Segmentation'][0]).astype(np.float32)

        return _img, _target


    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomResizedCrop(size=513),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
        tr.ToTensor()])

    sbd_train = SBDSegmentation(split='train',
                                transform=composed_transforms_tr)

    dataloader = DataLoader(sbd_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['gt'].numpy()
            gt[gt > 0.5] = 255
            gt = np.array(gt[jj]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img[jj][0], cmap='gray')
            plt.subplot(212)
            plt.imshow(gt[0], cmap='gray')

        if ii == 1:
            break
    plt.show(block=True)