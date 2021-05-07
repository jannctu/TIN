import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_img(dat,rows,columns,fname):
    w=10
    h=10
    #fig, axarr = plt.figure(figsize=(8, 8))
    d,_,_ = dat.shape
    print(d)
    counter = 0
    f, axarr = plt.subplots(rows, columns)
    for r in range(rows):
        for c in range(columns):
            axarr[r, c].imshow(dat[counter], cmap='gray')
            axarr[r, c].axis('off')
            counter = counter + 1
    plt.savefig(fname, bbox_inches='tight')


def init_model(model):
    shape = (3, 3)
    a = np.array([custom_kernel(shape,0),
                 custom_kernel(shape, 22.5),
                 custom_kernel(shape,45),
                 custom_kernel(shape, 67.5),
                 custom_kernel(shape,90),
                 custom_kernel(shape, 112.5),
                 custom_kernel(shape,135),
                 custom_kernel(shape, 157.5),
                 custom_kernel(shape,180),
                 custom_kernel(shape, 202.5),
                 custom_kernel(shape,225),
                 custom_kernel(shape, 247.5),
                 custom_kernel(shape,270),
                 custom_kernel(shape, 292.5),
                 custom_kernel(shape,315),
                 custom_kernel(shape, 337.5),])

    a = np.array([a, a, a])
    a = np.transpose(a, (1, 0, 2, 3))
    model.conv1_1.weight.data = torch.from_numpy(a).float()

def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
           for i in range(shape[1])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k


def custom_kernel(shape,deg):
    k = custom_sobel(shape, deg)
    kd = k
    for i in range(shape[0]):
        for j in range(shape[1]):
            kd[i, j] = np.cos(np.deg2rad(deg)) * k[i, j] + np.sin(np.deg2rad(deg)) * k[i, j]

    return kd



