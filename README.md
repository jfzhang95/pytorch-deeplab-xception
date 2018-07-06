# pytorch-deeplab-xception

### Introduction
This is a Pytorch implementation of [DeepLab-Xception](https://arxiv.org/pdf/1802.02611), It's based on a Modified Aligned Xception backbone. Currently, we train DeepLab V3+ from scratch, using Pascal Voc 2012 dataset.

### TODO
- [x] Basic deeplab v3+ model
- [x] Training deeplab v3+ on SBD dataset
- [ ] Add pretrained model and results evaluation


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



