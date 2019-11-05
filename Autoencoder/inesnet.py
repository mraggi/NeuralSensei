import torch
import torch.nn as nn
import fastai.vision as fv
import fastai.basics as fb
import torch.nn.functional as F

from Layers import *
from Mish import Mish

def InesNet(c_out):
    initial = nn.Sequential(NiceDownscale(3,64,k=6,activation=False))

    blockA = nn.Sequential(concat_downscale(64,128),
                           ResBlock(128),
                           NiceDownscale(128,256),
                           ResBlock(256,g=2),
                           ResBlock(256,bottle=192),
                          )
    
    blockB = nn.Sequential(concat_downscale(256,512),
                           ResBlock(512,bottle=256,g=4),
                           ResBlock(512,bottle=384,g=2),
                           ResBlock(512,g=4),
                           ResBlock(512,g=2),
                           concat_downscale(512,768),
                           ResBlock(768,bottle=512,g=4),
                           ResBlock(768,g=2),
                           ResBlock(768,bottle=384,g=3),
                           ResBlock(768))
                     
    
    classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                               fv.Flatten(),
                               Mish(),
                               nn.BatchNorm1d(768),
                               nn.Linear(768,c_out))
    
    return nn.Sequential(initial, blockA, blockB, classifier)
