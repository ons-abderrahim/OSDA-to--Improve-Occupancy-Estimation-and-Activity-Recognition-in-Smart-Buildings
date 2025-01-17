import torch
from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torch.autograd.variable import *
import math

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass
    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
    def output_num(self):
        pass


## Some classes from https://github.com/ksaito-ut/OPDA_BP/blob/master/models/basenet.py

resnet_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, "resnet101":models.resnet101, "resnet152":models.resnet152}

class ResNetFc(BaseFeatureExtractor):
    def __init__(self, model_name='resnet50', model_path=None, normalize=True):
        super(ResNetFc, self).__init__()

        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.__in_features = 9216########9216/32768



    def forward(self, x):
        #x = x.view(-1,1,x.size(1))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.conv5(out)
        out = self.bn5(out)
        
        out = self.relu(out)
        x = out.view(out.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class Net_CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Net_CLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
    def forward(self, x):
        x = self.fc(x)
        return x

class Net_CLS_C(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim, bias=True):
        super(Net_CLS_C, self).__init__()
        self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
        self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=bias)
        self.main = nn.Sequential(self.bottleneck,
                                  nn.Sequential(nn.BatchNorm1d(bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True),
                                                self.fc))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class Net_CLS_DC(nn.Module):
    def  __init__(self, in_dim, out_dim, bottle_neck_dim=None):
        super(Net_CLS_DC, self).__init__()
        if bottle_neck_dim is None:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
            )
        else:
            self.main = nn.Sequential(nn.Linear(in_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, bottle_neck_dim), nn.LeakyReLU(0.2, inplace=True), \
                                      nn.Linear(bottle_neck_dim, out_dim))
    def forward(self, x):
        for module in self.main.children():
            x = module(x)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
