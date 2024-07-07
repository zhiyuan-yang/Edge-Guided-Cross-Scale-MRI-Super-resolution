import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x
    
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class Grad_Decoder(nn.Module):
    def __init__(self, nf, n_blks=[2, 2, 2, 12, 8, 4]):
        super().__init__()
        block = functools.partial(ResidualBlock, nf=nf)
        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])
        
        self.conv_L1 = nn.Conv2d(2*nf, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[2])
        
        self.conv_x2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_x2 = make_layer(block, n_layers=n_blks[4])
        
        self.conv_x4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_x4 = make_layer(block, n_layers=n_blks[5])
        
        self.act = nn.ReLU(inplace=True)
        
        self.conv_tail = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)
        
    def forward(self, grad):
        grad_L3 = self.act(self.conv_L3(grad[2]))  #H/8,W/8
        grad_L3 = self.blk_L3(grad_L3)
        
        grad_L2 = self.act(self.conv_L2(grad_L3))
        grad_L2 = self.blk_L2(grad_L2)
        grad_L2_up = F.interpolate(grad_L2, scale_factor=2, mode='bilinear') #H/4,W/4
        
        grad_L1 = self.act(self.conv_L1(torch.cat([grad_L2_up, grad[2]], dim=1)))
        grad_L1 = self.blk_L1(grad_L1)
        
        
        grad_x1_up = F.interpolate(grad_L1, scale_factor=2, mode='bilinear') #H/2, W/2
        grad_x2 = self.act(self.conv_x2(grad_x1_up))
        grad_x2 = self.blk_x2(grad_x2)
        
        grad_x2_up = F.interpolate(grad_x2, scale_factor=2, mode='bilinear') #H,W
        grad_x4 = self.act(self.conv_x4(grad_x2_up))
        grad_x4 = self.blk_x4(grad_x4)
        
        out = self.conv_tail(grad_x4)
        
        return [grad_L1, grad_x2, grad_x4], out