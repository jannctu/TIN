import torch
from model import TIN
import os
from os.path import join
import numpy as np
from datasets import Data_Loader
from torch.utils.data import DataLoader
from PIL import Image
import scipy.io as io
import cv2
import time

weight_file = 'weights/TIN2.pth'

model = TIN(False,2)
model.cuda()
model.eval()

#load weight
checkpoint = torch.load(weight_file)
model.load_state_dict(checkpoint)

pytorch_total_params = sum(p.numel() for p in model.parameters())

test_dataset = Data_Loader(split="test",scale=[0.5, 1, 1.5])
test_loader = DataLoader(
    test_dataset, batch_size=1,
    num_workers=1, drop_last=True, shuffle=False)

save_dir = 'results2'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    if not os.path.exists(join(save_dir,"mat")):
        os.mkdir(join(save_dir,"mat"))
    if not os.path.exists(join(save_dir,"png")):
        os.mkdir(join(save_dir,"png"))

idx = 0
start_time = time.time()

with torch.no_grad():
    for i, (image, ori, img_files) in enumerate(test_loader):
        #print(ori.shape)
        h = ori.size()[1]
        w = ori.size()[2]
        ms_fuse = np.zeros((h,w))

        for img in image:
            #print(img.shape)
            img = img.cuda()
            out = model(img)
            fuse = out[-1].squeeze().detach().cpu().numpy()
            fuse = cv2.resize(fuse, (w, h), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse
        ms_fuse /= len(image)

        filename = img_files[0][0][5:-4]
        result = Image.fromarray((ms_fuse * 255).astype(np.uint8))
        result.save(join(save_dir,"png", "%s.png" % filename))
        io.savemat(join(save_dir,"mat", '{}.mat'.format(filename)), {'result': ms_fuse})
        print("Running test [%d/%d]" % (idx + 1, len(test_loader)))
        idx = idx + 1
    print('finished.')
print("--- %s seconds ---" % (time.time() - start_time))
print(pytorch_total_params)
