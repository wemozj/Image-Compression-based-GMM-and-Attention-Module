# from .basics import *
# import pickle
# import os
# import codecs
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class BitEstimator(nn.Module):
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        # x = torch.exp(x)  # add by wemo
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

class GMM_Module(nn.Module):
    """
    GMM Module
    """
    # def __init__(self, out_channel_M):
    #     super(GMM_Module, self).__init__()
    #     self.conv1 = nn.Conv2d(int(out_channel_M), 3 * out_channel_M, kernel_size=1)
    #     torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2 * 1 * (3+1)/(1+1)))
    #     torch.nn.init.constant_(self.conv1.bias.data, 0.01)
    #     self.lrelu_1 = nn.LeakyReLU()
    #
    #     self.conv2 = nn.Conv2d(3 * out_channel_M, 6 * out_channel_M, kernel_size=1)
    #     torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * 1 * (3+6)/(3+3)))
    #     torch.nn.init.constant_(self.conv2.bias.data, 0.01)
    #     self.lrelu_2 = nn.LeakyReLU()
    #
    #     self.conv3 = nn.Conv2d(6 * out_channel_M, 9 * out_channel_M, kernel_size=1)
    #     torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2 * 1 * (6 + 9) / (6 + 6)))
    #     torch.nn.init.constant_(self.conv3.bias.data, 0.01)
    
    def __init__(self, out_channel_M, k):
        super(GMM_Module, self).__init__()
        self.conv1 = nn.Conv2d(int(out_channel_M), k * out_channel_M, kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2 * 1 * (k + 1) / (1 + 1)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.lrelu_1 = nn.LeakyReLU()
    
        self.conv2 = nn.Conv2d(k * out_channel_M, 2*k * out_channel_M, kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * 1 * (k + 2*k) / (k + k)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.lrelu_2 = nn.LeakyReLU()
    
        self.conv3 = nn.Conv2d(2*k * out_channel_M, 3*k * out_channel_M, kernel_size=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2 * 1 * (2*k + 3*k) / (2*k + 2*k)))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
    

    def forward(self, input):
        x = self.lrelu_1(self.conv1(input))
        x = self.lrelu_2(self.conv2(x))
        return self.conv3(x)