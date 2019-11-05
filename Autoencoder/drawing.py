import fastai
import fastai.basics as fai
import fastai.vision as fv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv

import random

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects
import matplotlib as mpl
from pathlib import Path

from Layers import *
from ranger import *
from mxresnet import *
from Mish import *

from fastai.callbacks import *
from functools import partial

def torchimg2numpy(t):
    return np.transpose(t.detach().cpu().numpy(),(1,2,0))

def show_tensor_as_image(tensor, ncols=5, figsize=10, title = ""):
    plt.figure(figsize=(figsize,figsize))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(tv.utils.make_grid(tensor.detach().cpu()[:ncols*ncols], nrow=ncols, padding=2, normalize=True).cpu(),(1,2,0)))
    
def draw_img_boxes(im, boxes, txts, colors=None, bgcolors=None, grid=None):
    if type(im) == torch.Tensor:
        im = torchimg2numpy(im)
    if colors == None: colors = ['white']*len(boxes)
    if bgcolors == None: bgcolors = ['black']*len(boxes)
    fig,ax = plt.subplots(1)
    X,Y,_ = im.shape
    ax.imshow(im)
    for box, txt, color, bgcolor in zip(boxes,txts,colors,bgcolors):
        x,y,w,h = box
        x,y,w,h = [int(q) for q in (x*X, y*Y, w*X, h*Y)]
        rect = patches.Rectangle((y,x),h,w,linewidth=1,edgecolor=color,facecolor='none')
        ax.add_patch(rect)
        cent = patches.Circle((y+h/2,x+w/2),radius=2,facecolor='b')
        ax.add_patch(cent)
        txt = plt.text(y,x,txt,color=color,size=10)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground=bgcolor)])
        rect.set_path_effects([PathEffects.withStroke(linewidth=3, foreground=bgcolor)])
    if grid != None:
        ticks=np.linspace(0,X,grid+1)
        for t in ticks:
            ax.plot([t,t], [0,Y],color='r')
            ax.plot([0,X], [t,t],color='r')

    #ax.axis('off')
    plt.show()
    
def draw_with_bboxes(img_file, bboxes):
    im = np.array(Image.open(img_path/img_file), dtype=np.uint8)
    
    draw_img_boxes(im,bboxes[0],bboxes[1])
    
def draw_from_batch(x,bboxes,lbls,lbl2txt, i = 0,grid=None):
    im = x[i]
    bb = (bboxes[i]+1)/2
    bb[:,2:] -= bb[:,:2]
    lb = lbls[i]
    
    good_boxes = []
    good_labels = []
    
    for box,label in zip(bb,lb):
        if label != 0:
            good_boxes.append(box)
            good_labels.append(lbl2txt[label])
    
    draw_img_boxes(im,good_boxes,good_labels,grid=grid)
    
def show_img(batch,i,cat2txt,grid=None):
    imgs, targs = batch
    im = imgs[i]
    filtered = (targs[targs[:,0] == i])
    boxes = filtered[:,2:]
    cats = filtered[:,1].long()
    labels = [cat2txt[j.item()] for j in cats]
    draw_img_boxes(torchimg2numpy(im),boxes,labels,grid=grid)
    
