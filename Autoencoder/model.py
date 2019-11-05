import torch
import torch.nn as nn

import fastai.vision as vision
from Layers import *

class Restructure(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        bs,n = x.shape
        return x.view((bs,n//4,2,2))
        
def encoder(c_out):
    return nn.Sequential(
        NiceDownscale(3,32,k=6,activation=False),
        nn.MaxPool2d(2),
        concat_downscale(32,64),
        ResBlock(64),
        concat_downscale(64,128),
        ResBlock(128),
        concat_downscale(128,256),
        ResBlock(256),
        NiceDownscale(256,512),
        ResBlock(512),
        NiceDownscale(512,1024),
        fv.Flatten(),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,c_out),
        nn.Tanh()
    )

def decoder(c_in):
    return nn.Sequential(
        nn.BatchNorm1d(c_in),
        nn.Linear(c_in,1024),
        Restructure(), #2
        fv.PixelShuffle_ICNR(256,leaky=0.1),#4
        ResBlock(256),
        conv_block(256,512,k=1,s=1),
        fv.PixelShuffle_ICNR(512,256,leaky=0.1), #8
        ResBlock(256),
        conv_block(256,512,k=1,s=1),
        fv.PixelShuffle_ICNR(512,256,leaky=0.1), #16
        ResBlock(256),
        fv.PixelShuffle_ICNR(256,128,leaky=0.1), #32
        ResBlock(128),
        fv.PixelShuffle_ICNR(128,64,leaky=0.1,blur=True),#64
        ResBlock(64),
        fv.PixelShuffle_ICNR(64,32,leaky=0.1,blur=True), #128
        conv_block(32,3,k=1,s=1,pad=0),
        ResBlock(3),
        nn.Sigmoid()
    )

class AutoEncoder(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.encoder = encoder(nf)
        self.decoder = decoder(nf)
    def forward(self, x):
        z = self.encoder(x)
        
        w = z + torch.randn_like(z)*0.05 if self.training else z
        return self.decoder(w)
