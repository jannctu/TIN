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
    #for i in range(1, columns*rows +1):
    #for i in range(1,d+1):
        #img = np.random.randint(10, size=(h,w))
        #x = fig.add_subplot(rows, columns, i)
        #x.title.set_text((i-1)*22.5)
        #plt.imshow(x0, cmap='gray')
        #axarr[0, 1].axis('off')
        #plt.imshow(dat[i-1],cmap='gray')
    plt.savefig(fname, bbox_inches='tight')


def init_model(model):
    # b = np.array([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    #                [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
    #                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    #                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    #                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                [[0, -1, -2], [1, 0, -1], [2, 1, 0]]],
    #               [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    #                [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
    #                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    #                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    #                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                [[0, -1, -2], [1, 0, -1], [2, 1, 0]]],
    #               [[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    #                [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    #                [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
    #                [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    #                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    #                [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
    #                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    #                [[0, -1, -2], [1, 0, -1], [2, 1, 0]]]
    #               ])
    # print(b.shape)
    # aa = np.array([a,a,a])
    #shape = (3,3)
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
    #print(a.shape)
    #print(model.conv1_1.weight.data.shape)
    #quit()
    model.conv1_1.weight.data = torch.from_numpy(a).float()
    #plot_img(a[0])
    # a = np.array([[custom_kernel(shape,0)],
    #               [custom_kernel(shape,45)],
    #               [custom_kernel(shape,90)],
    #               [custom_kernel(shape,135)],
    #               [custom_kernel(shape,180)],
    #               [custom_kernel(shape,225)],
    #               [custom_kernel(shape,270)],
    #               [custom_kernel(shape,315)]],
    #              [[custom_kernel(shape,0)],
    #               [custom_kernel(shape,45)],
    #               [custom_kernel(shape,90)],
    #               [custom_kernel(shape,135)],
    #               [custom_kernel(shape,180)],
    #               [custom_kernel(shape,225)],
    #               [custom_kernel(shape,270)],
    #               [custom_kernel(shape,315)]],
    #               [[custom_kernel(shape, 0)],
    #                [custom_kernel(shape, 45)],
    #                [custom_kernel(shape, 90)],
    #                [custom_kernel(shape, 135)],
    #                [custom_kernel(shape, 180)],
    #                [custom_kernel(shape, 225)],
    #                [custom_kernel(shape, 270)],
    #                [custom_kernel(shape, 315)]]
    #              )

    #a = np.array([a, a, a])
    #a = np.expand_dims(a, axis=1)
    #aa = np.repeat(a, a, axis=0)
    #a = np.transpose(a, (1, 0, 2, 3))
    #print(a.shape)
    #print(model.conv1.weight.data.shape)
   # quit()
    #model.conv1_1.weight.data = torch.from_numpy(a).float()

    #x, y = np.mgrid[-5:6, -5:6]
    #gaussian_kernel = np.exp(-(x ** 2 + y ** 2))
    #print(model.conv2_1.weight.data.shape)
    # Normalization
    # gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    # gaussian_kernel = np.expand_dims(gaussian_kernel,axis=0)
    #print(gaussian_kernel.shape)
    #b = np.repeat(gaussian_kernel, 64, axis=0)
    #b = np.expand_dims(b, axis=0)
    #b = np.array([b, b, b])
    #b = np.repeat(b, 16, axis=0)

    #b = np.transpose(b, (1, 0, 2, 3))
    #print(b.shape)
    #print(gaussian_kernel)
    #plt.imshow(gaussian_kernel, cmap=plt.get_cmap('jet'), interpolation='nearest')
    #plt.colorbar()
    #plt.show()
    #model.conv2_1.weight.data = torch.from_numpy(b).float()

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

    return k



