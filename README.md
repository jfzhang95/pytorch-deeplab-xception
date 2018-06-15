# pytorch-deeplab-xception

## Introduction
This is a Pytorch implementation of [DeepLab-Xception](https://arxiv.org/pdf/1802.02611), It's based on a Modified Aligned Xception backbone. We train DeepLab V3+ from scratch, using Pascal Voc 2012 dataset.

## Status
In progress

## Requirements
```
Python3.x (Tested with 3.5)
PyTorch (Tested with 0.4.0)
tensorboardX
opencv-python
```

### Installation
To use this code, please do:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. To train DeepLabV3+, please do:
    ```Shell
    python train.py
    ```



