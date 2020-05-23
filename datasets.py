# borrowed from https://github.com/meteorshowers/RCF-pytorch

from torch.utils import data
from os.path import join
from PIL import Image
import numpy as np
import cv2
#from matplotlib import pyplot as plt

def prepare_image_cv2(im):
    im = cv2.resize(im, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
    im = np.transpose(im, (2, 0, 1))  # (H x W x C) to (C x H x W)
    return im


class Data_Loader(data.Dataset):
    def __init__(self, root='../DATA/data', split='train', scale=None):
        self.root = root
        self.split = split
        self.scale = scale
        self.bsds_root = join(root, 'HED-BSDS')
        if self.split == 'train':
            self.filelist = join(self.root, 'bsds_pascal_train_pair.lst')
        elif self.split == 'test':
            #self.filelist = join(self.bsds_root, 'image-test.lst')
            self.filelist = join(self.bsds_root, 'test_pair.lst')
        else:
            raise ValueError("Invalid split type!")
        with open(self.filelist, 'r') as f:
            self.filelist = f.readlines()

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        r = np.random.randint(0, 100000)
        if self.split == "train":
            img_file, lb_file = self.filelist[index].split()
            lb = np.array(Image.open(join(self.root, lb_file)), dtype=np.float32)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = cv2.resize(lb, (256, 256), interpolation=cv2.INTER_LINEAR)

            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 64)] = 2
            lb[lb >= 64] = 1
            # lb[lb >= 128] = 1
            img = np.array(cv2.imread(join(self.root, img_file)), dtype=np.float32)
            img = prepare_image_cv2(img)
            return img, lb
        else:
            img_file, lb_file = self.filelist[index].split()
            data = []
            data_name = []

            original_img = np.array(cv2.imread(join(self.bsds_root, img_file)), dtype=np.float32)
            img = cv2.resize(original_img, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

            if self.scale is not None:
                for scl in self.scale:
                    img_scale = cv2.resize(img, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
                    data.append(img_scale.transpose(2, 0, 1))
                    data_name.append(img_file)
                return data, img, data_name

            img = prepare_image_cv2(img)

            lb = np.array(Image.open(join(self.bsds_root, lb_file)), dtype=np.float32)

            if lb.ndim == 3:
                lb = np.squeeze(lb[:, :, 0])
            assert lb.ndim == 2
            lb = cv2.resize(lb, (256, 256), interpolation=cv2.INTER_LINEAR)
            lb = lb[np.newaxis, :, :]
            lb[lb == 0] = 0
            lb[np.logical_and(lb > 0, lb < 64)] = 2
            lb[lb >= 64] = 1

            return img, lb, img_file

