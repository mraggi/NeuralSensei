import fastai.basics as fai
import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from Mish import Mish
from mxresnet import *

def num_params(model):
    total=0
    for p in model.parameters():
        num=1
        for s in list(p.size()):
            num *= s
        total += num
    return total

def replace_module_by_other(model,module,other):
    for child_name, child in model.named_children():
        if isinstance(child, module):
            setattr(model, child_name, other)
        else:
            replace_module_by_other(child,module,other)

def conv_block(ni, no, k=3, s=1, pad="same", bn = True, activation = True, g = 1):
    if pad == "same": 
        pad = k//2
    
    layers = []
    
    if activation:
        layers += [nn.ReLU()]
    
    if bn:
        layers += [nn.BatchNorm2d(ni)]
       
    layers += [nn.Conv2d(ni, no, kernel_size=k, stride=s, padding=pad, groups=g)]
    
    return nn.Sequential(*layers)

class FastNormalize(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(1,nf,1,1))
    def forward(self, x):
        return self.gamma*x

class ResBlock(nn.Module):
    def __init__(self, nf, bottle=None, s=1, pre_activation=True, g=1):
        super().__init__()
        
        if bottle == None: bottle = nf
        
        self.act = nn.ReLU(inplace=True) if pre_activation else identity
        self.pre_bn = nn.BatchNorm2d(nf)
        
        BN = nn.BatchNorm2d(nf)
        nn.init.constant_(BN.weight, 0.)
        
        k = 3 if s == 1 else 4
        
        self.residual = nn.Sequential(nn.Conv2d(nf,bottle,kernel_size=1,stride=1,padding=0),
                                      nn.ReLU(),
                                      nn.BatchNorm2d(bottle),
                                      nn.Conv2d(bottle,nf,kernel_size=k,stride=s,padding=1,groups=g,bias=False),
                                      BN)
        self.pool = identity if s==1 else nn.AvgPool2d(2)
        
    def forward(self, x):
        y = self.pre_bn(self.act(x))
        return self.pool(y) + self.residual(y)


def identity(x):
    return x

def init_for_avg(conv, with_pixel_shuffle=False):
    k = conv.kernel_size[0]
    s = conv.stride[0]
    assert(k%2 == 0)
    with torch.no_grad():
        if conv.bias is not None: conv.bias.zero_()
        T = conv.weight
        
        no = T.shape[0]
        ni = T.shape[1]
        
        assert(no >= ni)
        a=(k-s)//2
        b=a+s
    
        T[:ni].zero_()
        for i in range(ni):
            T[i,i,a:b,a:b] = 1/s**2
        if with_pixel_shuffle:
            nf = ni+ni*k**2
            assert(no >= nf)
            T[ni:nf].zero_()
            for i in range(ni):
                for x in range(k):
                    for y in range(k):
                        j = i*k*k + x*k + y
                        T[ni+j,i,x,y]=1
        

def NiceDownscale(ni, no, g=1, k=4, s=2, activation=True, bn=True,with_pixel_shuffle=False):
    cb = conv_block(ni,no,k=k,s=s,pad=(k-1)//2,activation=activation,bn=bn)
    init_for_avg(cb[-1],with_pixel_shuffle)
    return cb

def pixel_unshuffle(x,r=2):
    b,c,h,w = x.shape
    out_channel = c*(r**2)
    out_h = h//r
    out_w = w//r
    fm_view = x.contiguous().view(b, c, out_h, r, out_w, r)
    return fm_view.permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w)

class PixelUnshuffle(nn.Module):
    def __init__(self,ratio=2):
        super().__init__()
        self.r = ratio
    def forward(self, x):
        return pixel_unshuffle(x,self.r)
    
class SplitAndMerge(nn.Module):
    def __init__(self, modelA, modelB):
        super().__init__()
        self.A = modelA
        self.B = modelB
        
    def forward(self, x):
        return torch.cat((self.A(x), self.B(x)),dim=1)

def concat_downscale(ni, no, g=1):
    assert(no > ni)
    path1 = ResBlock(ni,s=2,g=g)
    path2 = conv_block(ni,no-ni,k=4,s=2,pad=1)
    return SplitAndMerge(path1,path2)

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return fv.spectral_norm(conv)

class SimpleSelfAttention(nn.Module):
    
    def __init__(self, ni:int, ks=1, sym=False):#, n_out:int):
        super().__init__()
           
        self.conv = conv1d(ni, ni, ks, padding=ks//2, bias=False)      
       
        self.gamma = nn.Parameter(torch.tensor([0.]))
        
        self.sym = sym
        self.ni = ni
        
    def forward(self,x):
        
        
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.ni,self.ni)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.ni,self.ni,1)
                
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
          
        o = self.gamma * o + x
        
          
        return o.view(*size).contiguous()

class RandomResizeLayer(nn.Module):
    def __init__(self, min_size, max_size=None, stride=32, prob=0.75, diffxy=False):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.stride = stride
        self.prob = prob
        self.diffxy = diffxy
    
    def forward(self, x):
        if not self.training or random.random() > self.prob:
            return x
        
        max_sz = self.max_size
        if max_sz == None:
            max_sz = x.shape[2]
        if self.min_size >= max_sz:
            min_size = max_size
        rx = random.choice(range(self.min_size,max_sz+1,self.stride))
        ry = rx
        if self.diffxy:
            ry = random.choice(range(self.min_size,max_sz+1,self.stride))
        
        return F.interpolate(x, size=(rx,ry), mode='bicubic',align_corners=False)

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)
