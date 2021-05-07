import os, sys
from os.path import join
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from datasets import Data_Loader
from model import TIN
from logger import Logger
from utils import init_model
from torch.utils.tensorboard import SummaryWriter

"""
learning rate adjustment div by 10 every 10 epochs
"""
def adjust_learning_rate(optimizer, epoch):
    lr = 1e-2 * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def make_optimizer(model, lr):
    optim = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    return optim

def save_img_progress(results, filename):
    if not os.path.isdir('img_log'):
        os.mkdir('img_log')

    results_all = torch.zeros((len(results), 1, 256, 256))
    # print(results[0].shape)
    for i in range(len(results)):
        results_all[i, 0, :, :] = results[i][0]
    torchvision.utils.save_image(1 - results_all, join('img_log', "%s.jpg" % filename))

def save_ckpt(model, name):
    print('saving checkpoint ... {}'.format(name), flush=True)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    torch.save(model.state_dict(), os.path.join('checkpoints', '{}.pth'.format(name)))

"""
balance cross entropy 
"""
def balanced_cross_entropy_loss(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = torch.nn.functional.binary_cross_entropy(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost) / (num_negative + num_positive)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
data loader
"""
batch_size = 1
train_dataset = Data_Loader(split="train")
train_loader = DataLoader(
    train_dataset, batch_size=batch_size,
    num_workers=8, drop_last=True, shuffle=True)

log = Logger('log.txt')
sys.stdout = log

"""
create model
"""
model = TIN(False,2)
#conv1_w = model.conv1_1.weight.data
init_model(model)
#conv1_ww = model.conv1_1.weight.data
#print(conv1_w==conv1_ww)
model.cuda()
model.train()

"""
PARAMS
"""
init_lr = 1e-2
total_epoch = 120
#####
each_epoch_iter = len(train_loader)
total_iter = total_epoch * each_epoch_iter
# print(each_epoch_iter)
#####
print_cnt = 10
ckpt_cnt = 500
cnt = 0
avg_loss = 0.

writer = SummaryWriter()
optim = make_optimizer(model, init_lr)

print('*' * 60)
print('train images in all are %d ' % each_epoch_iter)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('total params in all are %d ' % pytorch_total_params)
print('*' * 60)

"""
START TRAINING
"""
for epoch in range(total_epoch):
    for i, (image, label) in enumerate(train_loader):
        cnt += 1
        if epoch % 10 == 0:
            adjust_learning_rate(optim, epoch)

        image, label = image.cuda(), label.cuda()
        outs = model(image)
        total_loss = 0

        for each in outs:
            total_loss += balanced_cross_entropy_loss(each, label)/batch_size

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        avg_loss += float(total_loss)
        if cnt % print_cnt == 0:
            writer.add_scalar('Loss/train', avg_loss / print_cnt, cnt)
            print('[{}/{}] loss:{} avg_loss: {}'.format(cnt, total_iter, float(total_loss), avg_loss / print_cnt),
                  flush=True)
            avg_loss = 0
            save_img_progress(outs, 'iter-{}'.format(cnt))

        if cnt % ckpt_cnt == 0:
            save_ckpt(model, 'weight-{}-iter-{}'.format(init_lr, cnt))

save_ckpt(model, 'final-model')