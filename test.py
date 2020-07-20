import torch
from model import TIN
import os
from os.path import join
import numpy as np
from PIL import Image
import scipy.io as io
import cv2
import time

test_img = 'img/mri_brain.jpg'
## READ IMAGE
im = np.array(cv2.imread(test_img), dtype=np.float32)
## Multiscale
scales = [0.5,1.0,1.5]
images = []
for scl in scales:
    img_scale = cv2.resize(im, None, fx=scl, fy=scl, interpolation=cv2.INTER_LINEAR)
    images.append(img_scale.transpose(2, 0, 1)) # (H x W x C) to (C x H x W)

## CREATE MODEL
weight_file = 'weights/TIN2.pth'
model = TIN(False,2)
model.cuda()
model.eval()
#load weight
checkpoint = torch.load(weight_file)
model.load_state_dict(checkpoint)

## FEED FORWARD
h, w, _ = im.shape
ms_fuse = np.zeros((h, w))

with torch.no_grad():
    for img in images:
        img = img[np.newaxis, :, :, :]
        img = torch.from_numpy(img)
        img = img.cuda()
        out = model(img)
        fuse = out[-1].squeeze().detach().cpu().numpy()
        fuse = cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)
        ms_fuse += fuse
    ms_fuse /= len(scales)

    filename = 'mri_brain'
    result = Image.fromarray(255-(ms_fuse * 255).astype(np.uint8))
    result.save( "img/result_%s.png" % filename)
print('finished.')

