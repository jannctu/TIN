import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Enrichment(nn.Module):
    def __init__(self, c_in, rate=2):
        super(Enrichment, self).__init__()
        self.rate = rate
        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        dilation = self.rate * 4 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu4 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        o4 = self.relu3(self.conv4(o))
        out = o + o1 + o2 + o3 + o4
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class TIN(nn.Module):
    def __init__(self,pretrain=False,tin_m=2):
        super(TIN, self).__init__()
        self.tin_m = tin_m
        ## CONV stage 1
        self.conv1_1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1)

        self.em1_1 = Enrichment(16, 4)
        self.em1_2 = Enrichment(16, 4)
            # CONV DOWN
        self.conv1_1_down = nn.Conv2d(32, 8, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(32, 8, 1, padding=0)
            # SCORE
        self.score_stage1 = nn.Conv2d(8, 1, 1)
        if tin_m > 1:
            ## CONV stage 2
            self.conv2_1 = nn.Conv2d(16, 64, 3, padding=1)
            self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)

            self.em2_1 = Enrichment(64, 4)
            self.em2_2 = Enrichment(64, 4)
                # CONV DOWN
            self.conv2_1_down = nn.Conv2d(32, 8, 1, padding=0)
            self.conv2_2_down = nn.Conv2d(32, 8, 1, padding=0)
                # SCORE
            self.score_stage2 = nn.Conv2d(8, 1, 1)

        # RELU
        self.relu = nn.ReLU()
        if tin_m > 1:
            # POOL
            self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # pooling biasa
            self.score_final = nn.Conv2d(2, 1, 1)

        if pretrain:
            state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    print('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    print('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        if 'BN' in name:
                            param.zero_()
                        else:
                            param.normal_(0, 0.01)
        else:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        # =========================================
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        # =========================================
        conv1_1_down = self.conv1_1_down(self.em1_1(conv1_1))
        conv1_2_down = self.conv1_2_down(self.em1_2(conv1_2))
        # =========================================
        o1_out = self.score_stage1(conv1_1_down + conv1_2_down)
        # =========================================
        if self.tin_m > 1:
            pool1 = self.maxpool(conv1_2)
            # =========================================
            conv2_1 = self.relu(self.conv2_1(pool1))
            conv2_2 = self.relu(self.conv2_2(conv2_1))
            # =========================================
            conv2_1_down = self.conv2_1_down(self.em2_1(conv2_1))
            conv2_2_down = self.conv2_2_down(self.em2_2(conv2_2))
            # =========================================
            o2_out = self.score_stage2(conv2_1_down + conv2_2_down)
            # =========================================
            upsample2 = nn.UpsamplingBilinear2d(size=(h, w))(o2_out)
            # =========================================
            fuseout = torch.cat((o1_out, upsample2), dim=1)
            fuse = self.score_final(fuseout)
            results = [o1_out, upsample2, fuse]
            results = [torch.sigmoid(r) for r in results]
        else:
            results = [o1_out]
            results = [torch.sigmoid(r) for r in results]
        return results