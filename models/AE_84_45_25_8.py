# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:02:00 2022

@author: sameer_poudel
"""

import torch
from torch.nn import Linear, ReLU, LeakyReLU, MSELoss, Sequential, Module, BatchNorm1d
from torch.optim import Adam
from torchinfo import summary
from torch.nn import functional as F
from torch import nn
    
def nonlinear_transform(in_features, out_features):
    
    transform =  Sequential(
                 Linear(in_features, out_features),
                 #BatchNorm1d(out_features),
                 LeakyReLU(0.01, inplace=True)    
                 )
    return transform

class AE(Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation            
        self.hidden1 = nonlinear_transform(84, 45)
        self.hidden2 = nonlinear_transform(45, 25)
        self.hidden3 = nonlinear_transform(25, 8)
        self.hidden4 = nonlinear_transform(8, 25)
        self.hidden5 = nonlinear_transform(25, 45)
        self.output = Linear(45, 84)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x1 = self.hidden1(x)
        #print(x.shape)
        x2 = self.hidden2(x1)
        #print(x.shape)
        x3 = self.hidden3(x2)
        #print(x.shape)
        x4 = self.hidden4(x3)
        #print(x.shape)
        x5 = self.hidden5(x4)
        #print(x.shape)
        x6 = self.output(x5)
        #print(x.shape)
        return x3, x6
            
model = AE().double()

#"""
if __name__ == "__main__":
    
    image = torch.rand((32,84))
    model = AE()
    print(model(image))
    print(summary(model, (32,84)))
#_""" 
