import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, sbd, combine_dbs
from dataloaders import utils
from networks import deeplab_xception
from dataloaders import custom_transforms as tr


gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False # Whether to use SBD dataset
nEpochs = 300  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 6  # Training batch size
testBatch = 6  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 50  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-6  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 40 # How many epochs to change learning rate

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    run_id = 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Network definition
net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=21, pretrained=True)
modelName = 'deeplabv3+'
criterion = utils.cross_entropy2d

if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomResizedCrop(size=512, scale=(0.5, 1.0)),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=512),
        tr.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
        tr.ToTensor()])

    voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = voc_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=2)

    utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['image'], sample_batched['gt']
            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            output = net.forward(inputs)
            # output = upsample(output, size=(513, 513), mode='bilinear', align_corners=True)

            loss = criterion(output, gts, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")


            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            net.eval()
            for ii, sample_batched in enumerate(testloader):
                inputs, gts = sample_batched['image'], sample_batched['gt']

                # Forward pass of the mini-batch
                inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
                if gpu_id >= 0:
                    inputs, gts = inputs.cuda(), gts.cuda()

                with torch.no_grad():
                    output = net.forward(inputs)
                # output = upsample(output, size=(513, 513), mode='bilinear', align_corners=True)

                loss = criterion(output, gts, size_average=False, batch_average=True)
                running_loss_ts += loss.item()

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:
                    running_loss_ts = running_loss_ts / num_img_ts
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    print('Loss: %f' % running_loss_ts)
                    running_loss_ts = 0


    writer.close()
