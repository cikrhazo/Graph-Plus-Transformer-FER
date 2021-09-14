import torch
import torch.utils.data as data
import torch.optim as optim
from network import GraphViT
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from data_utilz.oulu import OuluSet
from data_utilz.graphs import Graph

import time
import logging
import os, sys
import argparse

from utilz import str2bool
import numpy as np
import random, math, copy
from torch.autograd import Variable
import torch.nn as nn
from termcolor import colored
from criteria import CenterLoss

from mmcv.utils import get_logger


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def load_GPUS(model, state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    return model


def load_state_dict(model, state_dict_path):

    pre_trained_dict = torch.load(state_dict_path)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pre_trained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pre_trained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def model_parameters(_structure, _parameterDir):

    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint
    model_state_dict = _structure.state_dict()
    # 1. filter out unnecessary keys
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    pretrained_state_dict.pop('mlp_head.1.weight', None)
    pretrained_state_dict.pop('mlp_head.1.bias', None)
    pretrained_state_dict.pop('to_latent.1.weight', None)
    pretrained_state_dict.pop('to_latent.1.bias', None)
    # 2. overwrite entries in the existing state dict
    model_state_dict.update(pretrained_state_dict)
    _structure.load_state_dict(model_state_dict)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train_val(args):

    beginner = args.beginner
    stride = args.stride
    device = args.device
    batch_size = args.batch_size
    epoch = args.Epoch
    lr = args.lr
    weight_decay = args.weight_decay
    model_path = "./models/"
    num_class = args.num_class
    img_size = args.image_size
    window = args.window
    valid = 0

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join("./logs/", f'{timestamp}.log')
    logger = get_logger(name="Oulu-Graph-Trans", log_file=log_file, log_level=logging.INFO)
    logger.info(f"batch size {batch_size}")

    # training
    TrainSet = OuluSet(valid=valid, train=True, out_size=img_size, window_size=window, online_ldm=False)
    TrainLoader = data.DataLoader(TrainSet, batch_size=batch_size, shuffle=True,
                                  pin_memory=False, num_workers=8, collate_fn=collate_fn)

    iters = int(TrainSet.__len__() / batch_size * (epoch - beginner))
    logger.info("Validation Folder: ")

    # validation
    ValidSet = OuluSet(valid=valid, train=False, out_size=img_size, window_size=window, online_ldm=False)
    ValidLoader = data.DataLoader(ValidSet, batch_size=16, shuffle=False,
                                  pin_memory=False, num_workers=8, collate_fn=collate_fn)
    logger.info(ValidSet.valid_sub)

    A = torch.from_numpy(Graph().A).float()
    net = GraphViT(
        dim=512,
        depth=3,
        heads=8,
        mlp_dim=512,
        num_classes=num_class,
        A=A,
        pool="mean"
    )

    if args.load_checkpoint:
        print("Loading check point")
        net.load_state_dict(torch.load("./models/oulu%d.pth" % (valid+1)))
    else:
        print("Loading pretrained model")
        model_parameters(net, "./pre_train/pre_trained.pth")

    optimizer = Adam(
        net.parameters(),
        weight_decay=weight_decay, lr=lr
    )

    net = nn.DataParallel(net, device_ids=[1, 0])
    net.cuda(device=device)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss().cuda(device)

    regular = CenterLoss(num_classes=num_class, feat_dim=2 * 512).cuda(device)
    optim_center = Adam(regular.parameters(), lr=1e-2)

    tmp = filter(lambda x: x.requires_grad, net.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)
    print("# Training Samples: " + str(TrainSet.__len__()))

    BestAcc = 0.
    for i in range(beginner, epoch):
        cls_loss = 0
        trainSamples = 0
        numCorrTrain = 0
        lam = 0
        net.train(True)

        for batch_idx, (geo_tensor, vis_tensor, emo_tensor) in enumerate(TrainLoader):

            trainSamples += emo_tensor.size(0)
            optimizer.zero_grad()

            geo_tensor = Variable(geo_tensor.cuda(device=device), requires_grad=False)
            vis_tensor = Variable(vis_tensor.cuda(device=device), requires_grad=False)
            emo_tensor = Variable(emo_tensor.cuda(device=device), requires_grad=False)
            
            feat, out = net(vis_tensor, geo_tensor)
            loss_ce = criterion(out, emo_tensor.squeeze())
            loss_re = regular(feat, emo_tensor.squeeze())
            loss = loss_ce + 0.001 * loss_re  #
            
#             # (Trick) Todo: analysis on this trick
#             geo_, vis_, y_vis, y_geo, hyper = trick(geo_tensor, vis_tensor, emo_tensor)
#             geo_ = Variable(geo_.cuda(device=device), requires_grad=False)
#             vis_ = Variable(vis_.cuda(device=device), requires_grad=False)
#             y_vis = Variable(y_vis.cuda(device=device), requires_grad=False)
#             y_geo = Variable(y_geo.cuda(device=device), requires_grad=False)

#             feat, out = net(vis_, geo_)
#             loss_ce = tricks_criterion(criterion, out, y_vis.squeeze(), y_geo.squeeze(), hyper)
#             loss_re = tricks_criterion(regular, feat, y_vis.squeeze(), y_geo.squeeze(), hyper)
#             loss = loss_ce + 0.001 * loss_re

            loss.backward()
            optimizer.step()
            optim_center.step()

            scheduler.step()
            # sched_center.step()

            cls_loss += loss * geo_tensor.size(0)
            if batch_idx % 10 == 9 or batch_idx == 0:
                print('#batch: %3d; loss_cls/cen: (%3.4f, %3.4f); learning rate: (%.4e, %.4e); Lam: %.4e'
                      % (batch_idx + 1, loss_ce, loss_re, optimizer.param_groups[0]['lr'],
                         optim_center.param_groups[0]['lr'], lam))

        avg_cls = cls_loss / trainSamples
        logger.info('Train: Epoch = %3d | CLS Loss = %.4f | Train Samples = %3d;'
                    % (i + 1, avg_cls, trainSamples))

        if i % stride == (stride - 1):
            torch.save(net.module.state_dict(), os.path.join(model_path, 'net_epoch_' + str(i + 1).zfill(3) + '.pth'))

        # Validation
        net.train(False)
        validSamples = 0
        numCorrValid = 0
        for batch_idx, (geo_valid, vis_valid, emo_valid) in enumerate(ValidLoader):
            geo_valid = Variable(geo_valid.cuda(device), requires_grad=False)
            vis_valid = Variable(vis_valid.cuda(device), requires_grad=False)
            emo_valid = Variable(emo_valid.cuda(device), requires_grad=False)
            with torch.no_grad():
                _, logits = net(vis_valid, geo_valid)
            label_t = emo_valid.detach().squeeze()
            _, label_p = torch.max(logits.data, 1)
            numCorrValid += (label_p == label_t.squeeze()).sum()
            validSamples += emo_valid.size(0)

        validAccuracy = (int(numCorrValid) / validSamples) * 100

        if BestAcc <= validAccuracy:
            BestAcc = validAccuracy
            torch.save(net.module.state_dict(), os.path.join(model_path, 'Oulu%d.pth' % (valid+1)))
        logger.info(
            'Train: Epoch = %3d | Valid Samples = %3d' % (i + 1, validSamples)
            + colored(' | Oulu Accuracy = %.4f', 'red') % validAccuracy
            + colored(' | Best Oulu = %.4f', 'green') % BestAcc
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--image_size", type=int, default=224, help="width and height should be identical")
    parser.add_argument("--image_channel", type=int, default=3)
    parser.add_argument("--num_frame", type=int, default=16)
    parser.add_argument("--stride", type=int, default=50, help='the stride for saving models')
    parser.add_argument("--window", type=int, default=49, help="# local patch size")
    parser.add_argument("--num_class", type=int, default=6, help="# of the classes")

    # For Training
    parser.add_argument("--load_checkpoint", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beginner", type=int, default=0)
    parser.add_argument('--Epoch', type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    train_val(args)
