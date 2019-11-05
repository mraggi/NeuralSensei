import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QCheckBox, QGridLayout, QHBoxLayout, QGroupBox, QMenu, QPushButton, QRadioButton, QVBoxLayout, QWidget, QSlider, QLabel)
from PyQt5.QtGui import QImage, QPixmap

import fastai.vision as fv
import random
import torch
import numpy as np

from model import *

def torch2qt(t):
    img = np.transpose(t.detach().cpu().numpy(),(1,2,0))*255
    img8 = img.astype(np.uint8, order='C', casting='unsafe')
    #print(img8)
    height, width, channel = img8.shape
    bytesPerLine = 3 * width
    return QImage(img8.data, width, height, bytesPerLine, QImage.Format_RGB888)

class Window(QWidget):
    def __init__(self, z, decoder, parent=None):
        super(Window, self).__init__(parent)
        
        self.setWindowTitle("Autoencoder")
        self.resize(1000, 600)
        self.z = z
        self.decoder = decoder
        self.img = self.resetImg()
        
        horizontal = QHBoxLayout()
        grid = QGridLayout()
        
        n = 10
        self.n = n
        w = 0
        for i in range(n):
            for j in range(n):
                grid.addWidget(self.createSlider(i,j,z[w].item()), i, j)
                w += 1
        
        self.lb = QLabel(self)
        self.pixmap = QPixmap.fromImage(self.img)
        self.pixmap = self.pixmap.scaled(512,512)
        self.lb.setPixmap(self.pixmap)
        horizontal.addWidget(self.lb)
        horizontal.addLayout(grid)
        
        self.setLayout(horizontal)

        
    
    def resetImg(self):
        return torch2qt(self.decoder(self.z[None,:]).squeeze())
        
    def OnSliderMove(self,i,j,value):
        self.z[i*self.n+j] = value/100
        
        self.img = self.resetImg()
        self.pixmap = QPixmap.fromImage(self.img)
        self.pixmap = self.pixmap.scaled(512,512)
        self.lb.setPixmap(self.pixmap)
        

    def createSlider(self,i,j, value):
        groupBox = QGroupBox(f"{i}{j}")
        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(1)
        slider.setValue(int(value*100))
        slider.setMaximum(100)
        slider.setMinimum(-100)
        slider.valueChanged.connect(lambda x: self.OnSliderMove(i,j,x))

        vbox = QVBoxLayout()
        vbox.addWidget(slider)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox

if __name__ == '__main__':
    model = AutoEncoder(100)
    model.eval()
    fv.requires_grad(model,False)
    model.load_state_dict(torch.load("perceptual6.pth",map_location=torch.device('cpu'))['model'])
    img1 = fv.open_image("david.jpg")
    img1.resize(128)
    img2 = fv.open_image("ica.jpg")
    img2.resize(128)
    img3 = fv.open_image("niko.png")
    img3.resize(128)
    
    z1 = model.encoder(img1.data.unsqueeze(0))
    z2 = model.encoder(img2.data.unsqueeze(0))
    z3 = model.encoder(img3.data.unsqueeze(0))
    #z = torch.rand(100).unsqueeze(0)*2-1
    #z#2 = model.encoder(img2.data.unsqueeze(0))
    #print(f"z = {z.shape}")
    #z = torch.rand((100,))*2-1
    z = (z1+z2+z3)/3
    QApplication.setStyle("Windows")
    app = QApplication(sys.argv)
    win = Window(z.squeeze(), model.decoder)
    win.show()
    sys.exit(app.exec_())
