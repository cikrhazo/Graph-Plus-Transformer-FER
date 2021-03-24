import numpy as np
import torch.utils.data as data
import face_alignment
import cv2
import os, random
import torch
import matplotlib.pyplot as plt
import logging
from data_utilz.utilz import multi_frame, visual_aggregation

emotion_list = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', "Contempt"]


class CK(data.Dataset):
    '''
        1: Surprise  0
        2: Fear      1
        3: Disgust   2
        4: Happiness 3
        5: Sadness   4
        6: Anger     5
        7: Contempt  6
        '''
    def __init__(self, root="/media/ruizhao/programs/datasets/Face/CK+/CK+_Video/",
                 out_size=224, window_size=49, online_ldm=False, num_frame=16, valid=0, train=True):

        self.root = root
        self.root_ldm = "/media/ruizhao/programs/datasets/Face/CK+/CK+_Video_ldm/"
        self.win_size = window_size
        self.out_size = out_size
        self.online_ldm = online_ldm
        self.paths = []
        self.num_frame = num_frame
        self.train = train

        #  Thank Adrian Bulat for the implementation at https://github.com/1adrianb/face-alignment
        if online_ldm:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda:1', flip_input=False)
        else:
            self.fa = None
        for _, dir_subjects, _ in os.walk(root):
            dir_subjects.sort()
            folder_sub = len(dir_subjects) // 10
            if valid == 9:
                self.valid_sub = dir_subjects[valid * folder_sub::]
            else:
                self.valid_sub = dir_subjects[valid * folder_sub: (valid+1) * folder_sub]
            self.train_sub = list(set(dir_subjects).difference(set(self.valid_sub)))
            self.train_sub.sort()
            self.valid_sub.sort()
            if train:
                dir_sub = self.train_sub
            else:
                dir_sub = self.valid_sub
            for subject in dir_sub:
                sub_root = os.path.join(root, subject)
                for _, dir_emotions, _ in os.walk(sub_root):
                    dir_emotions.sort()
                    for emotion in dir_emotions:
                        emotion_root = os.path.join(sub_root, emotion)
                        self.paths.append(emotion_root)
                    break
            break

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        label_ = path.split('/')[-1]
        label = emotion_list.index(label_)
        if label is None:
            logging.info('Label Error')
            logging.error('Error: CANNOT Find Label from Sequence: {}!'.format(path))
            raise ValueError()
        geo_input = np.empty(shape=(self.num_frame, 51, 2))
        vis_input = np.empty(shape=(51, 3, self.win_size, self.win_size, self.num_frame))
        for _, _, files in os.walk(path):
            files.sort()
            if self.train:
                num_select = random.randint(12, len(files))
                idxes = np.linspace(0, num_select - 1, self.num_frame).tolist()
                files = [files[int(i)] for i in idxes]
                self.flip = random.sample([True, False], 1)[0]
            else:
                self.flip = False
            for ii in range(len(files)):
                file = files[ii]
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1] / 255.
                img = cv2.resize(img, dsize=(self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)
                if self.flip and self.online_ldm:
                    img = np.fliplr(img)
                if self.online_ldm:
                    with torch.no_grad():
                        lmarks = self.fa.get_landmarks(np.uint8(img * 255))  # a list of 68 point: [(x_0, y_0), (x_1, y_1), ...]
                        lmarks = np.array(lmarks)[:, 17:, :]
                    if lmarks is None:
                        logging.info('Landmark Error')
                        logging.error('Error: CANNOT Find Landmarks from Sequence: {}!'.format(path))
                        raise ValueError()
                else:
                    landmark_path = img_path.split('/')[-3:]
                    landmark_path = ['/'+ii.replace(".png", ".txt") for ii in landmark_path]
                    landmark_path = "".join(landmark_path)[1:]
                    landmark_file = open(os.path.join(self.root_ldm, landmark_path), "r")
                    lmarks = landmark_file.readlines()
                    lmarks = [[float(lmarks[i].split(" ")[0]), float(lmarks[i].split(" ")[1])] for i in range(68)]
                    lmarks = np.array(lmarks)[np.newaxis, 17:, :]
                if self.train:
                    sigma = random.uniform(0, 5)
                    lmarks = lmarks + np.random.randn(*lmarks.shape) * sigma
                    sigma = random.uniform(0, 10)
                    img = img + np.random.randn(*img.shape) * (sigma / 255.)

                visual = visual_aggregation(img, lmarks[0], window_size=self.win_size)  # 68*wz*wz*3
                landmarks = lmarks[0]
                visualize = visual.transpose((0, 3, 1, 2))  # 51*3*wz*wz

                # plt.figure(0)
                # plt.imshow(img.clip(0, 1))
                # plt.title(label)
                # for k in range(51):
                #     plt.scatter(x=landmarks[k, 0], y=landmarks[k, 1])
                #     plt.text(x=landmarks[k, 0], y=landmarks[k, 1], s=str(k))
                # plt.show()

                vis_input[:, :, :, :, ii] = visualize  # 51*3*wz*wz*5
                geo_input[ii, :, :] = landmarks

            break

        geo_input = multi_frame(geo_input).transpose((0, 3, 1, 2))  # 3, T, P, C -> 3, C, T, P
        geo_input = torch.from_numpy(geo_input.astype(np.float32))  # 3, C, T, P
        vis_input = torch.from_numpy(vis_input.astype(np.float32))  # P, Ch, H, W, T
        emo_label = torch.Tensor([label]).long()
        return geo_input, vis_input, emo_label


if __name__ == "__main__":
    for valid in range(0, 10):
        ValidSet = CK(root="/media/ruizhao/programs/datasets/Face/CK+/CK+_Video/",
                           valid=valid, train=True, online_ldm=False)
        ValidLoader = data.DataLoader(ValidSet, batch_size=32, num_workers=8, shuffle=False, pin_memory=False)
        print(ValidSet.__len__())
        print(ValidSet.valid_sub)
        for batch_idx, (tensor_geo, tensor_vis, tensor_tar) in enumerate(ValidLoader):
            print(tensor_geo.size())

