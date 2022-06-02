import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np
#import hsv

import copy
from typing import Optional, Any

import numpy as np
import torch
from torch import nn,Tensor
import torch.nn.functional as F
import torch as t

from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout

class TF(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.(PWY)

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TF(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> srcc = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super(TF, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.relu = nn.ReLU(inplace=True)
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TF, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        b,c,s=src.permute(1,2,0).size()
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                               key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers
    


class enhance_net_nopool(nn.Module):

    def __init__(self):
        super(enhance_net_nopool, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        number_f = 4
        self.e_conv1 = nn.Conv2d(1,1,3,1,1,1,groups = 1,bias=True)
        self.pw1 = nn.Conv2d(1,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,1,groups = number_f,bias=True) 
        self.pw2 = nn.Conv2d(number_f,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,1,groups = number_f,bias=True)
        self.pw3 = nn.Conv2d(number_f,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,1,groups = number_f,bias=True) 
        self.pw4 = nn.Conv2d(number_f,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2,number_f*2,3,1,1,1,groups = number_f*2,bias=True) 
        self.pw5 = nn.Conv2d(number_f*2,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2,number_f*2,3,1,1,1,groups = number_f*2,bias=True)
        self.pw6 = nn.Conv2d(number_f*2,number_f,1,1,0,1,groups = 1,bias=True)
        self.e_conv7 = nn.Conv2d(number_f*2,number_f*2,3,1,1,1,groups = number_f*2,bias=True)
        self.pw7 = nn.Conv2d(number_f*2,1,1,1,0,1,groups = 1,bias=True)
        
        # self.model = nn.Sequential(
        #
        #     nn.Conv2d(1, 16 , 3, stride=2, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.InstanceNorm2d(16, affine=True),
        #     *discriminator_block(16, 32,normalization=True),
        #     #*discriminator_block(32, 64,normalization=True),
        #     #*discriminator_block(64, 128, normalization=True),
        #     *discriminator_block(32, 32),
        #     #*discriminator_block(128, 128, normalization=True),
        #     #nn.Dropout(p=0.5),
        #     nn.Conv2d(32, 2, 8, padding=0),
        # )
        self.model = nn.Sequential(

            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
        )
        self.TF = TF(16, 8)
        self.F = nn.Conv2d(16, 2, 16, padding=0)


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        #self.level_esti_net1 = nn.Linear(128, 256)
        #self.level_esti_net2 = nn.Linear(256, 128)
        #self.level_esti_net3 = nn.Linear(256, 64)
        self.adapt = nn.AdaptiveAvgPool2d((32,32))



        
    def forward(self, x):
        batch,w,h,b = x.shape
        #rth = hsv.RGB_HSV().cuda()
        red,green,blue = torch.split(x ,1,dim = 1)
        #h,s,v = rth.rgb_to_hsv(x)
        v = (red + green + blue)/3
        #eh = rth.GBlur(h,1)
        #es = rth.GBlur(s,1)
        x1 = self.e_conv1(v)
        p1 = self.relu(self.pw1(x1))
        x2 = self.e_conv2(p1)
        p2 = self.relu(self.pw2(x2))
        x3 = self.e_conv3(p2)
        p3 = self.relu(self.pw3(x3))
        x4 = self.e_conv4(p3)
        p4 = self.relu(self.pw4(x4))
        x5 = self.e_conv5(torch.cat([p3,p4],1))
        p5 = self.relu(self.pw5(x5))
        # x5 = self.upsample(x5)
        x6 = self.e_conv6(torch.cat([p2,p5],1))
        p6 = self.relu(self.pw6(x6))
        x7 = self.e_conv7(torch.cat([p1,p6],1))
        v_r = F.sigmoid(self.pw7(x7))
        #DCE_net = model1.enhance_net_nopool().cuda()
        #DCE_net.load_state_dict(torch.load('snapshots1/Epoch49.pth'))
        #x_r= DCE_net(x)
        zero = 0.000001*torch.ones_like(v)
        one = 0.999999*torch.ones_like(v)
        v0 = torch.where(v>0.999999,one,v)
        v0 = torch.where(v<0.000001,zero,v0)
        r = v_r


        # v64 = F.interpolate(v,size=(64,64),mode='nearest')
        # level = F.sigmoid(self.model(v64))
        # #level7 = torch.ones_like(level6)
        # #level = torch.cat([level6,level7],1)
        v32 = F.interpolate(v,size=(32,32),mode='nearest')
        
        v1 = self.model(v32)
        bb, cc, ww, hh = v1.size()
        v2 = v1.view(bb,cc,-1).permute(2, 0, 1)
        v3 = self.TF(v2)
        v4 = v3.permute(1,2,0)
        v5 = v4.view(bb,cc,ww,hh)
        level = F.sigmoid(self.F(v5))
        

        
        #g0 = torch.sigmoid(10*g2)
        #b0 = torch.sigmoid(b2)
        #level = torch.cat([g0,b0],1)
        #level = ([[0.25,0.08]])
        
        g1 = level[0,0].item()
        b1 = level[0,1].item()
        #g1 = 0.25
        #b1 = 0.08


        g = 0.1*g1+0.18
        b = 0.04*b1+0.06
        #b = 0.04

        for i in range(batch):
            if(i == 0):
                r0 = torch.pow (0.1*level[i,0].item()+0.18,torch.unsqueeze(r[i,:,:,:],0))
                #r0 = torch.pow (0.4,torch.unsqueeze(r[i,:,:,:],0))
            else:
                r1 = torch.pow (0.1*level[i,0].item()+0.18,torch.unsqueeze(r[i,:,:,:],0))
                #r1 = torch.pow (0.4,torch.unsqueeze(r[i,:,:,:],0))
                r0 = torch.cat([r0,r1],0)

        ev = torch.pow(v0,r0)
        #ev1 = rth.GBlur(ev,5)
        
        #L = 0.3 - 7*x
        for i in range(batch):
            if(i == 0):
                #L = 0.5*torch.pow(0.04*level[i,1],0.5*level[i,0].item())*(1-(1/0.06*level[i,1])*(torch.unsqueeze(x0[i,:,:,:],0)))
                L = 400*torch.pow((0.04*level[i,1].item()+0.06 - torch.unsqueeze(v[i,:,:,:],0)),3)
                #L = 400*torch.pow((0.08 - torch.unsqueeze(v[i,:,:,:],0)),3)

            else:
                #L0 = 0.5*torch.pow(0.06*level[i,1],0.5*level[i,0].item())*(1-(1/0.06*level[i,1])*(torch.unsqueeze(x0[i,:,:,:],0)))
                L0 = 400*torch.pow((0.04*level[i,1].item()+0.06 - torch.unsqueeze(v[i,:,:,:],0)),3)
                #L0 = 400*torch.pow((0.08 - torch.unsqueeze(v[i,:,:,:],0)),3)
                L = torch.cat([L,L0],0)

        L = torch.where(L<0.00001,zero,L)

        ev = ev - L
        #hsv1 = torch.cat([h,s,ev],1)
        #enhance_image = rth.hsv_to_rgb(hsv1)
        #enhance_image = torch.where(x<0.000001,zero,enhance_image)
        v = v + 0.000001
        red1 = red/v
        green1 = green/v
        blue1 = blue/v
        red0 = red1*ev
        green0 = green1*ev
        blue0 = blue1*ev
        enhance_image = torch.cat([red0,green0,blue0],1)
        zero1 = torch.zeros_like(x)
        vvv = torch.cat([v,v,v],1)
        t = torch.where(vvv>0.04,zero1,enhance_image)
        A = r
        enhance_image_1 = enhance_image
        return enhance_image_1,enhance_image,A,g,b,t



