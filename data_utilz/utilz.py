import numpy as np
import math
import argparse
import random, torch
import cv2


def multi_input(data):
    P, C = data.shape  # 68 2

    data_new = np.zeros((3, P, C))
    # normalization
    eye_c1 = np.mean(data[36:42, :], axis=0)
    eye_c2 = np.mean(data[42:48, :], axis=0)

    dis = math.sqrt(((eye_c2 - eye_c1) ** 2).sum()) * 2
    data = data / dis

    face_c = np.mean(data[0: 17, :], axis=0)
    eyebrow_c1 = np.mean(data[17: 22, :], axis=0)
    eyebrow_c2 = np.mean(data[22: 27, :], axis=0)
    nose_c = np.mean(data[27: 36, :], axis=0)
    eye_c1 = np.mean(data[36: 42, :], axis=0)
    eye_c2 = np.mean(data[42: 48, :], axis=0)
    lips_c = np.mean(data[48: 60, :], axis=0)
    teeth_c = np.mean(data[60: 68, :], axis=0)

    data_new[0, :, :] = data  # A
    for i in range(P):
        data_new[1, i, :] = data[i, :] - data[33, :]  # R
        if i in range(0, 17):
            data_new[2, i, ] = data[i, :] - face_c
        elif i in range(17, 22):
            data_new[2, i, ] = data[i, :] - eyebrow_c1
        elif i in range(22, 27):
            data_new[2, i, ] = data[i, :] - eyebrow_c2
        elif i in range(27, 36):
            data_new[2, i, ] = data[i, :] - nose_c
        elif i in range(36, 42):
            data_new[2, i, ] = data[i, :] - eye_c1
        elif i in range(42, 48):
            data_new[2, i, ] = data[i, :] - eye_c2
        elif i in range(48, 60):
            data_new[2, i, ] = data[i, :] - lips_c
        elif i in range(60, 68):
            data_new[2, i, ] = data[i, :] - teeth_c

    return data_new


def visual_aggregation(img, landmarks, window_size=25):
    stride = int(window_size / 2) + 1
    img = np.pad(img, ((window_size, window_size), (window_size, window_size), (0, 0)), mode='constant')
    visual = np.zeros(shape=(landmarks.shape[0], window_size, window_size, 3))  # 3 is the channels (RGB)
    for i in range(landmarks.shape[0]):
        x, y = int(landmarks[i, 0]), int(landmarks[i, 1])
        crop = img[y+stride:y+stride+window_size, x+stride:x+stride+window_size, :]  # because of padding
        if crop.shape[0] != window_size:
            pad = window_size-(crop.shape[0] % window_size)
            crop = np.pad(crop, ((pad, 0), (0, 0), (0, 0)), mode='constant')
        if crop.shape[1] != window_size:
            pad = window_size-(crop.shape[1] % window_size)
            crop = np.pad(crop, ((0, 0), (pad, 0), (0, 0)), mode='constant')
        visual[i, :, :, :] = crop
    return visual


def multi_frame(data):
    T, P, C = data.shape  # 16, 51 2

    data_new = np.zeros((3, T, P, C))

    # normalization
    for ii in range(T):
        data[ii, :, :] = data[ii, :, :] - data[ii, 16, :]  # R
        eye_c1 = np.mean(data[ii, 19: 25, :], axis=0)
        eye_c2 = np.mean(data[ii, 25: 31, :], axis=0)

        dis = math.sqrt(((eye_c2 - eye_c1) ** 2).sum()) * 2
        data[ii, :, :] = data[ii, :, :] / dis

        data_new[0, ii, :, :] = data[ii, :, :]  # relative

        # face_c = np.mean(data[ii, 0: 17, :], axis=0)
        eyebrow_c1 = np.mean(data[ii, 0: 5, :], axis=0)
        eyebrow_c2 = np.mean(data[ii, 5: 10, :], axis=0)
        nose_c = np.mean(data[ii, 10: 19, :], axis=0)
        eye_c1 = np.mean(data[ii, 19: 25, :], axis=0)
        eye_c2 = np.mean(data[ii, 25: 31, :], axis=0)
        lips_c = np.mean(data[ii, 31: 43, :], axis=0)
        teeth_c = np.mean(data[ii, 43: 51, :], axis=0)

        # component
        for i in range(P):
            # if i in range(0, 17):
            #     data_new[1, ii, i, ] = data[ii, i, :] - face_c
            if i in range(0, 5):
                data_new[1, ii, i, ] = data[ii, i, :] - eyebrow_c1
            elif i in range(5, 10):
                data_new[1, ii, i, ] = data[ii, i, :] - eyebrow_c2
            elif i in range(10, 19):
                data_new[1, ii, i, ] = data[ii, i, :] - nose_c
            elif i in range(19, 25):
                data_new[1, ii, i, ] = data[ii, i, :] - eye_c1
            elif i in range(25, 31):
                data_new[1, ii, i, ] = data[ii, i, :] - eye_c2
            elif i in range(31, 43):
                data_new[1, ii, i, ] = data[ii, i, :] - lips_c
            elif i in range(43, 51):
                data_new[1, ii, i, ] = data[ii, i, :] - teeth_c
    # temporal
    for j in range(T):
        if j > 0:
            data_new[2, j, :, :] = data[j, :, :] - data[j-1, :, :]
    return data_new


def random_erase(image, area_ratio_range=0.05, min_aspect_ratio=0.3, max_attempt=100):
    # image = np.asarray(image).copy()
    #
    # if np.random.random() > p:
    #     return image

    sl, sh = area_ratio_range, area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    h, w = image.shape[:2]
    image_area = h * w

    for _ in range(max_attempt):
        mask_area = np.random.uniform(sl, sh) * image_area
        aspect_ratio = np.random.uniform(rl, rh)
        mask_h = int(np.sqrt(mask_area * aspect_ratio))
        mask_w = int(np.sqrt(mask_area / aspect_ratio))

        if mask_w < w and mask_h < h:
            x0 = np.random.randint(0, w - mask_w)
            y0 = np.random.randint(0, h - mask_h)
            x1 = x0 + mask_w
            y1 = y0 + mask_h
            image[y0:y1, x0:x1] = np.zeros(shape=(mask_h, mask_w, 3))
            break

    return image
