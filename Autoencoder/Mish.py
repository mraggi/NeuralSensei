import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x*torch.tanh(F.softplus(x))
