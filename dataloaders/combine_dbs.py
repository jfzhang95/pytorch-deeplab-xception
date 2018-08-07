import torch.utils.data as data


class CombineDBs(data.Dataset):
    def __init__(self, dataloaders, excluded=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.cat_list = []
        self.im_list = []
        new_im_ids = []
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (curr_im_id not in new_im_ids):
                    num_images += 1
                    new_im_ids.append(curr_im_id)
                    self.cat_list.append({'db_ii': ii, 'cat_ii': jj})

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}'.format(num_images))

    def __getitem__(self, index):

        _db_ii = self.cat_list[index]["db_ii"]
        _cat_ii = self.cat_list[index]['cat_ii']
        sample = self.dataloaders[_db_ii].__getitem__(_cat_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.cat_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:'+str(include_db)+'\n'+'Excluded datasets:'+str(exclude_db)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders import pascal
    from dataloaders import sbd
    import torch
    import numpy as np
    import dataloaders.custom_transforms as tr
    from dataloaders.utils import decode_segmap
    from torchvision import transforms

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(512, 512)),
        tr.ToTensor()])

    pascal_voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
    pascal_voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)

    dataset = CombineDBs([pascal_voc_train, sbd], excluded=[pascal_voc_val])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            tmp = np.squeeze(tmp, axis=0)
            segmap = decode_segmap(tmp)
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break
    plt.show(block=True)