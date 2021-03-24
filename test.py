import torch
import torch.utils.data as data
from network import GraphViT
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import scipy.io as scio

from data_utilz.oulu import OuluSet
from data_utilz.eNTERFace import eNTERFace
from data_utilz.ck import CK
from data_utilz.graphs import Graph
import argparse

import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from termcolor import colored


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    csfont = {'fontname': 'serif'}
    cm = cm * 100
    # plt.rcParams["font.family"] = "Times"
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=18, weight="bold", **csfont)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=18, weight="bold")
    plt.yticks(tick_marks, classes, fontsize=18, weight="bold")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=18, weight="bold")

    plt.ylabel('Ground-truth', fontsize=20, weight="bold")
    plt.xlabel('Prediction', fontsize=20, weight="bold")
    plt.tight_layout()


def test(args):
    class_name = ['Su', 'Fe', 'Di', 'Ha', 'Sa', 'An']  # 'Co'
    device = args.device
    num_class = args.num_class
    img_size = args.image_size
    window = args.window

    AllCorrValid = 0
    AllNumSample = 0

    for fold in range(5):
        print("Validation Folder: ")
        # validation
        # ValidSet = OuluSet(valid=fold, train=False, out_size=img_size, window_size=window, online_ldm=False)
        # ValidSet = CK(valid=fold, train=False, out_size=img_size, window_size=window, online_ldm=False)
        ValidSet = eNTERFace(valid=fold, train=False, out_size=img_size, window_size=window, online_ldm=False)
        ValidLoader = data.DataLoader(ValidSet, batch_size=17, shuffle=False, pin_memory=False, num_workers=8)
        print(ValidSet.valid_sub)

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

        net.load_state_dict(torch.load("./models/eNTERFace/eNTERFace%d.pth" % (fold + 1)))

        net = nn.DataParallel(net, device_ids=[1, 0])
        net.cuda(device=device)

        # Validation
        net.eval()
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
            if batch_idx == 0 and fold == 0:
                all_predicted = label_p
                all_targets = label_t
            else:
                all_predicted = torch.cat((all_predicted, label_p), 0)
                all_targets = torch.cat((all_targets, label_t), 0)
            numCorrValid += (label_p == label_t.squeeze()).sum()
            AllCorrValid += (label_p == label_t.squeeze()).sum()
            validSamples += emo_valid.size(0)
            AllNumSample += emo_valid.size(0)

        validAccuracy = (int(numCorrValid) / validSamples) * 100
        print(
            'Valid Fold = %d' % (fold + 1) + colored(' | Accuracy = %.4f', 'red') % validAccuracy
        )

    AllAccuracy = AllCorrValid / AllNumSample * 100
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    scio.savemat('./confusion_matrix/' + 'eNTERFace_gt.mat',
                 {'gt': all_targets.data.cpu().numpy()})
    scio.savemat('./confusion_matrix/' + 'eNTERFace_pd.mat',
                 {'pd': all_predicted.cpu().numpy()})
    print("Accuracy: %.2f" % AllAccuracy)
    np.set_printoptions(precision=4)
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(matrix, classes=class_name, normalize=True,
                          title='eNTERFace05: %.2f%%' % AllAccuracy)
    plt.savefig('./confusion_matrix/eNTERFace_cm.png')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For Dataset and Record
    parser.add_argument("--image_size", type=int, default=224, help="width and height should be identical")
    parser.add_argument("--image_channel", type=int, default=3)
    parser.add_argument("--num_frame", type=int, default=16)
    parser.add_argument("--window", type=int, default=49, help="# local patch size")
    parser.add_argument("--num_class", type=int, default=6, help="# of the classes")

    # For Training
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    test(args)
