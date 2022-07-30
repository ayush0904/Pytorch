'''
Activation Function apply a non-linear transformation and decide whether a neuron should be activated or not
Without activation function our network is basically just a stacked linear regression model
With non linear transformation our network can learn better and perform more complex tasks
After each layer we typically use an activation function

Activation Function : Step Function, Sigmoid, TanH, ReLU, Leaky ReLU, Softmax
Leaky ReLU is improved version of ReLU. Tries to solve the vanishing gradient problem,
multiply a very small values for -ve numbers 

For relu the -ve values is 0 so the gradient later in backpropagation will also be 0. 
Then weights will never be updated, hence the neurons will be dead.
'''

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F # Leaky Relu is only available here F.leaky_relu()
# option 1 (create nn modules)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out



# option 1 (use activation functions directly in forward pass)

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,1)

    
    def forward(self,x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out





