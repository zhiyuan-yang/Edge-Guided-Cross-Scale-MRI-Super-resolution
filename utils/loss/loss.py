from . import discriminator
from math import sqrt,log
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReconstructionLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ReconstructionLoss, self).__init__()
        if (type == 'l1'):
            self.loss = nn.L1Loss()
        elif (type == 'l2'):
            self.loss = nn.MSELoss()
        else:
            raise SystemExit('Error: no such type of ReconstructionLoss!')

    def forward(self, sr, hr):
        return self.loss(sr, hr)


class Get_gradient(nn.Module):
    def __init__(self):
        super().__init__()
        h = 1/(4 * sqrt(2 * log(2)))
        sigma = 2*h/0.01
        size = 3
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
        kernel /= np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data= kernel, requires_grad=False).cuda()
    
    def forward(self, x):
        x_v = F.conv2d(x, self.weight, padding=1)
        return x_v 
        
        
class StructLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.get_gradient = Get_gradient()
        self.loss = nn.MSELoss()
        
    def forward(self, struct_sr, hr):
        #hr_grad = self.get_gradient(hr)
        loss = self.loss(struct_sr, hr)
        return loss


def get_loss_dict(args, logger):
    loss = {}
    if (abs(args.rec_w - 0) <= 1e-8):
        raise SystemExit('NotImplementError: ReconstructionLoss must exist!')
    else:
        loss['rec_loss'] = ReconstructionLoss(type='l1')
    if  (abs(args.struct_w - 0) > 1e-8):
        loss['struct_loss'] = StructLoss()
    return loss