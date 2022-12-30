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
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation            
        self.hidden1 = nonlinear_transform(91, 45)
        self.hidden2 = nonlinear_transform(45, 25)
        # Output layer, 8 units - one for each digit
        self.output = Linear(25, 9)
  
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.output(x)
        
        return x

model = Network().double()

#"""
if __name__ == "__main__":
    
    image = torch.rand((32,91))
    model = Network()
    print(model(image))
    print(summary(model, (32,91)))
#_""" 
