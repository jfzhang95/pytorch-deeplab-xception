# pytorch-deeplab-xception

### Introduction
This is a Pytorch implementation of [DeepLab-Xception](https://arxiv.org/pdf/1802.02611), It's based on a Modified Aligned Xception backbone. Currently, we train DeepLab V3+ from scratch, using Pascal Voc 2012 dataset.

![Results](doc/results.png)

We use deeplab v3+ model trained on Pascal VOC 2012 and SBD datasets to inference these results.
After 50 training epoch, our deeplab v3+ model can reach 72.7% mIoU on Pascal VOC 2012 test set.

### Requirements
```
Python3.x (Tested with 3.5)
PyTorch (Tested with 0.4.0)
tensorboardX
opencv-python
```

### Installation
This code is tested in Ubuntu 16.04. To use this code, please do:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. To train DeepLabV3+, please do:
    ```Shell
    python train.py
    ```

### TODO
- [x] Basic deeplab v3+ model
- [x] Training deeplab v3+ on SBD dataset
- [x] Results evaluation on Pascal VOC 2012 test set
- [ ] Deeplab v3+ model using resnet as backbone
- [ ] Training deeplab v3+ on other datasets


