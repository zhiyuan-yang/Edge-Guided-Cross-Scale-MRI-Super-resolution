from mmcv.ops import DeformConv2dPack as DCN
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from math import sqrt,log
from . import Grad_Decoder


class DCAT(nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int = 64,
            num_heads: int = 8,
            dropout_rate: float = 0.1,
            pos_embed=True,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        # self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = DCA(input_size=input_size, input_size1=input_size,hidden_size=hidden_size, proj_size=proj_size, num_heads=num_heads, channel_attn_drop=dropout_rate,spatial_attn_drop=dropout_rate)

        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))
        else:
            self.pos_embed = None
            
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=1,stride=1),
        )

    def forward(self, x,ref):
        
        B, C, H, W = ref.shape
        ref = ref.reshape(B, C, H * W).permute(0, 2, 1)
        
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
            ref = ref + self.pos_embed
        attn = x + self.epa_block(self.norm(x),self.norm(ref))

        attn_skip = attn.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        attn_skip = self.ffn(attn_skip) + attn_skip
        return attn_skip

class DCA(nn.Module):

    def __init__(self, input_size,input_size1, hidden_size, proj_size, num_heads=4, qkv_bias=True,
                 channel_attn_drop=0.1, spatial_attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Linear(hidden_size,hidden_size)
        
        self.kvv = nn.Linear(hidden_size,hidden_size*3)
        
        self.E = self.F = nn.Linear(input_size1, proj_size)

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

        self.out_proj = nn.Linear(hidden_size, int(hidden_size // 2))
        self.out_proj2 = nn.Linear(hidden_size, int(hidden_size // 2))

    def forward(self, x,ref):
        B, N, C = x.shape
        B1,N1,C1 = ref.shape
        
        x = self.q(x)
        q_shared = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        kvv = self.kvv(ref).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        kvv = kvv.permute(2, 0, 3, 1, 4)
        k_shared, v_CA, v_SA = kvv[0], kvv[1], kvv[2]

        #### 通道注意力
        q_shared = q_shared.transpose(-2, -1) #B,Head,C,N
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature # (B,Head,C,N) * (#B,Head,N,C) -> (B,Head,C,C)

        attn_CA = attn_CA.softmax(dim=-1) #B,Head,C,C
        attn_CA = self.attn_drop(attn_CA)

        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,C,C) * (B,Head,C,N) -> (B,Head,C,N) -> (B,N,C)
        
        
        #### 位置注意力
        k_shared_projected = self.E(k_shared)
        v_SA_projected = self.F(v_SA)

        attn_SA = (q_shared.permute(0, 1, 3, 2) @ k_shared_projected) * self.temperature2 # (B,Head,N,C) * (B,Head,C,64) -> (B,Head,N,64)

        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)

        x_SA = (attn_SA @ v_SA_projected.transpose(-2, -1)).permute(0, 3, 1, 2).reshape(B, N, C) # (B,Head,N,64) * (B,Head,64,C) -> (B,Head,N,C) -> (B,N,C)

        # Concat fusion
        x_CA = self.out_proj(x_CA)
        x_SA = self.out_proj2(x_SA)
        x = torch.cat((x_SA, x_CA), dim=-1)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


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


class SAM(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(SAM, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv_shared = nn.Sequential(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            style = self.conv_shared(torch.cat([lr, ref], dim=1))
            gamma = self.conv_gamma(style)
            beta = self.conv_beta(style)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out


class mySAM(nn.Module):
    def __init__(self, nf) -> None:
        super(mySAM, self).__init__()
        self.conv_mu = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        self.conv_sigma = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        block = functools.partial(ResidualBlock, nf=nf)
        self.conv = nn.Conv2d(2*nf, nf, 3, 1,1)
        self.fuse = make_layer(block=block, n_layers=1)
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)
        
        
    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        sigma = self.conv_sigma(lr)
        mu = self.conv_mu(lr)
        
        fuse = ref_normed * mu + sigma
        fuse = torch.cat([fuse, lr], dim=1)
        fuse = self.fuse(self.conv(fuse))
        
        return fuse
            
    
class Encoder(nn.Module):
    def __init__(self, in_chl, nf, n_blks=[1, 1, 1], act='relu'):
        super(Encoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[2])

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))
        fea_L2 = self.blk_L2(self.act(self.conv_L2(fea_L1)))
        fea_L3 = self.blk_L3(self.act(self.conv_L3(fea_L2)))

        return [fea_L1, fea_L2, fea_L3]


class channel_attention(nn.Module):
    def __init__(self, nf, reduction_ration=16):
        super().__init__()
        self.mlp = nn.Sequential(
                                nn.Flatten(start_dim=1, end_dim=3),
                                nn.Linear(in_features=2*nf, out_features=2*nf // reduction_ration),
                                nn.ReLU(),
                                nn.Linear(in_features=2*nf // reduction_ration, out_features=2*nf)
                                )
    def forward(self, fused_fea):
        W = fused_fea.size(2)
        H = fused_fea.size(3)
        gamma = self.mlp(F.avg_pool2d(fused_fea, (W,H), stride=(W,H))) + self.mlp(F.max_pool2d(fused_fea, (W,H), stride=(W,H)))
        gamma = gamma.reshape(fused_fea.shape[0], fused_fea.shape[1], 1, 1)
        fused_fea = fused_fea * gamma
        
        return fused_fea
        

class EdgeEnhance(nn.Module):
    def __init__(self, nf):
        super().__init__()
        
        self.conv3_1 = nn.Conv2d(nf, nf, (3, 1), padding=(1, 0))
        self.conv1_3 = nn.Conv2d(nf, nf, (1,3), padding=(0, 1))
        self.conv1_1 = nn.Conv2d(2*nf, nf, 1, 1, 0)
        self.CA = channel_attention(nf=nf, reduction_ration=16)
        self.conv_tail = nn.Sequential(
                                        nn.Conv2d(2*nf, nf, 1, 1),
                                        nn.Conv2d(nf, nf, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nf, nf, 3, 1, 1)
                                      )
    def forward(self, fea, edge):
        out_re = fea
        fea_ver = self.conv3_1(fea)
        fea_hor = self.conv1_3(fea)
        fea_edge = self.conv1_1(torch.cat([fea_ver, fea_hor], dim=1))
        out = self.CA(torch.cat([fea_edge, edge], dim=1))
        out = self.conv_tail(out)
        
        return out + out_re
        
        


class Decoder(nn.Module):
    def __init__(self, nf, out_chl, n_blks=[1, 1, 1, 1, 1, 1]):
        super(Decoder, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)

        self.conv_L3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.blk_L3 = make_layer(block, n_layers=n_blks[0])

        self.conv_L2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.blk_L2 = make_layer(block, n_layers=n_blks[1])

        self.conv_L1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks[2])

        self.merge_warp_x1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x1 = make_layer(block, n_blks[3])
        self.edgeEnhance_x1 = EdgeEnhance(nf)

        self.merge_warp_x2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x2 = make_layer(block, n_blks[4])
        self.edgeEnhance_x2 = EdgeEnhance(nf)
        
        self.merge_warp_x4 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.blk_x4 = make_layer(block, n_blks[5])
        self.edgeEnhance_x3 = EdgeEnhance(nf)

        self.conv_out = nn.Conv2d(64, out_chl, 3, 1, 1, bias=True)

        self.act = nn.ReLU(inplace=True)

        #self.pAda = SAM(nf, use_residual=True, learnable=True)
        self.pAda = mySAM(nf)
        
    def forward(self, lr_l, warp_ref_l, grad):
        fea_L3 = self.act(self.conv_L3(lr_l[2])) #H/8,W/8
        fea_L3 = self.blk_L3(fea_L3)

        fea_L2 = self.act(self.conv_L2(fea_L3))
        fea_L2 = self.blk_L2(fea_L2)
        fea_L2_up = F.interpolate(fea_L2, scale_factor=2, mode='bilinear', align_corners=False) #H/4,W/4

        fea_L1 = self.act(self.conv_L1(torch.cat([fea_L2_up, lr_l[2]], dim=1)))
        fea_L1 = self.blk_L1(fea_L1)

        warp_ref_x1 = self.pAda(fea_L1, warp_ref_l[2])
        fea_x1 = self.act(self.merge_warp_x1(torch.cat([warp_ref_x1, fea_L1], dim=1)))
        fea_x1 = self.blk_x1(fea_x1)
        fea_x1 = self.edgeEnhance_x1(fea_x1, grad[0])
        fea_x1_up = F.interpolate(fea_x1, scale_factor=2, mode='bilinear', align_corners=False) #H/2,W/2

        warp_ref_x2 = self.pAda(fea_x1_up, warp_ref_l[1])
        fea_x2 = self.act(self.merge_warp_x2(torch.cat([warp_ref_x2, fea_x1_up], dim=1)))
        fea_x2 = self.blk_x2(fea_x2)
        fea_x2 = self.edgeEnhance_x2(fea_x2, grad[1])
        fea_x2_up = F.interpolate(fea_x2, scale_factor=2, mode='bilinear', align_corners=False)

        warp_ref_x4 = self.pAda(fea_x2_up, warp_ref_l[0])
        fea_x4 = self.act(self.merge_warp_x4(torch.cat([warp_ref_x4, fea_x2_up], dim=1)))
        fea_x4 = self.blk_x4(fea_x4)
        fea_x4 = self.edgeEnhance_x3(fea_x4, grad[2])
        out = self.conv_out(fea_x4)

        return out


class Get_gradient(nn.Module):
    def __init__(self, sigma, kernel_size):
        super().__init__()
        
        kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)/2)**2 + (y-(kernel_size-1)/2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
        kernel /= np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.cuda()
        self.weight = nn.Parameter(data= kernel, requires_grad=False).cuda()
    
    def forward(self, x):
        x_v = F.conv2d(x, self.weight, padding=1)
        return x_v
    

class csfm(nn.Module):
    def __init__(self, nf, reduction_ration=16):
        super().__init__()
        
        self.get_offset = torch.nn.Conv2d(2*nf, nf, kernel_size=1, stride=1, padding=0)
        self.dcpack = DCN(nf, nf, kernel_size=(3,3), stride=(1,1), padding=1, deform_groups=8)
        self.relu = nn.ReLU(inplace=True)
        self.CA = channel_attention(nf=nf, reduction_ration=16)
        
        self.convtail = nn.Conv2d(2*nf, nf, kernel_size=1, stride=1, padding=0)
        
    def forward(self, lr, lr_down):
        lr_up = F.interpolate(lr_down, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.get_offset(torch.cat([lr, lr_up], dim=1))
        lr_down_align = self.relu(self.dcpack([lr_up, offset]))
        fused_fea = torch.cat([lr_down_align, lr], dim=1)
        fused_fea = self.CA(fused_fea)
        fused_fea = self.convtail(fused_fea)
        
        return fused_fea
    
class DCAMSR(nn.Module):
    def __init__(self,scale):
        super().__init__()
        #baseling model parameters
        input_size = 240
        in_chl = 1
        nf = 64
        n_blks = [4, 4, 4]
        n_blks_dec = [2, 2, 2, 8, 8, 4]
        self.scale = scale
        depths = [1,1,1]
        
        #cross-scale fuse module parameter
        reduction_ratio = 16
        
        #self-similarity constrain parameter
        h = 1/(4 * sqrt(2 * log(2)))
        sigma = 2*h/0.01
        kernel_size = 3

        self.enc = Encoder(in_chl=in_chl, nf=nf, n_blks=n_blks)
        self.decoder = Decoder(nf, in_chl, n_blks=n_blks_dec)

        self.trans_lv1 = nn.ModuleList([DCAT(input_size=input_size*input_size, hidden_size=64, proj_size=64, pos_embed=i!=0) for i in range(depths[0])] )
        self.trans_lv2 = nn.ModuleList([DCAT(input_size=input_size*input_size//4, hidden_size=64, proj_size=64, pos_embed=i!=0) for i in range(depths[1])] )
        self.trans_lv3 = nn.ModuleList([DCAT(input_size=input_size*input_size//16, hidden_size=64, proj_size=64, pos_embed=i!=0)  for i in range(depths[2])] )

        self.csfm = csfm(nf, reduction_ration=reduction_ratio)
        
        self.get_gradient = Get_gradient(sigma=sigma, kernel_size=kernel_size)
        
        self.grad_decoder = Grad_Decoder.Grad_Decoder(nf, n_blks_dec)
        
    def forward(self, lr, ref, lr_gradient):        
        
        lrsr = F.interpolate(lr, scale_factor=self.scale, mode='bilinear')
        
        #lr_gradient = self.get_gradient(lr)
        
        lrsr_gradient = F.interpolate(lr_gradient, scale_factor=self.scale, mode='bilinear')
        
        fea_lrsr= self.enc(lrsr)
        fea_ref_l = self.enc(ref)
        gradient_fea_lrsr = self.enc(lrsr_gradient)
        
        fea_lrsr_fuse_x4 = self.csfm(fea_lrsr[0], fea_lrsr[1])
        
        fea_lrsr_fuse_x2 = self.csfm(fea_lrsr[1], fea_lrsr[2])
        
        fea_lrsr_fuse_x1 = fea_lrsr[2]
        
        for transformer in self.trans_lv1:
            warp_ref_patches_x4 = transformer(fea_lrsr_fuse_x4,fea_ref_l[0])
            
        for transformer in self.trans_lv2:
            warp_ref_patches_x2 = transformer(fea_lrsr_fuse_x2,fea_ref_l[1])
            
        for transformer in self.trans_lv3:
            warp_ref_patches_x1 = transformer(fea_lrsr_fuse_x1,fea_ref_l[2])

        warp_ref_l = [warp_ref_patches_x4, warp_ref_patches_x2, warp_ref_patches_x1]
        grad_x, out_grad = self.grad_decoder(gradient_fea_lrsr)
        out = self.decoder(fea_lrsr, warp_ref_l, grad_x)
        out = out + lrsr

        return out, out_grad