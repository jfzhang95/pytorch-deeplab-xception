import os

import torch, cv2
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if sample.ndim == 2:
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution, interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution, interpolation=flagval)

    return sample

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    logit = logit.contiguous().view(-1, c)
    target = target.view(-1)
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)