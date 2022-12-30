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
        self.hidden1 = nonlinear_transform(84, 60)
        self.hidden2 = nonlinear_transform(60, 40)
        self.hidden3 = nonlinear_transform(40, 20)
        self.hidden4 = nonlinear_transform(20, 6)
        self.hidden5 = nonlinear_transform(6, 20)
        self.hidden6 = nonlinear_transform(20, 40)
        self.hidden7 = nonlinear_transform(40, 60)
        self.output = Linear(60, 84)
        
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
        x6 = self.hidden6(x5)
        #print(x.shape)
        x7 = self.hidden7(x6)
        #print(x.shape)
        x8 = self.output(x7)
        #print(x.shape)
        return x4, x8
            
model = AE().double()

#"""
if __name__ == "__main__":
    
    image = torch.rand((32,84))
    model = AE()
    print(model(image))
    print(summary(model, (32,84)))
#_""" 
