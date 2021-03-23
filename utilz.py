import argparse
import numpy as np
import cv2


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'TRUE'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
