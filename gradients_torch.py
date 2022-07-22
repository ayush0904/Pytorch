# 1) Design model (input, output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - Forward Pass : Compute Prediction
#     - Backward Pass : Gradients
#     - Update Weights


# Scratch
import torch
import torch.nn as nn

from gradients_numpy import forward


# X = torch.tensor([1,2,3,4], dtype=torch.float32) 
# Y = torch.tensor([2,4,6,8], dtype=torch.float32) 

#w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

# model prediction
# Manually Forward
# def forward(x):
#     return w * x

X = torch.tensor([ [1],[2],[3],[4] ], dtype=torch.float32) 
Y = torch.tensor([ [2],[4],[6],[8] ], dtype=torch.float32) 

X_test = torch.tensor([5],dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples,n_features)

input_size = n_features
output_size = n_features

# Linear Model from nn library
#model = nn.Linear(input_size,output_size)

# This is the way to design a Pytorch Model 
class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        #Define Layers
        self.lin = nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

#Loss = MSE (Mean Square Error)

#  Manually define Loss
# def loss(y,y_predicted):
#     return ((y_predicted-y)**2).mean()



# print(f'Prediction Before Training: f(5) = {forward(5):.3f}')
print(f'Prediction Before Training: f(5) = {model(X_test).item():.3f}')

# Training 
learning_rate = 0.01
n_iters = 100

# MSE from nn library
loss = nn.MSELoss()
# STOCHASTIC GRADIENT DESCENT
#optimizer = torch.optim.SGD([w], lr=learning_rate)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    #prediction = forward pass
    # y_pred = forward(X)
    y_pred = model(X)

    # Loss 
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward() # dl/dw



    # update weights
    # this operation should not be a part of gradiwnt tracking (computational part)
    # Manually Update Weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad

    #Update Weights
    optimizer.step()

    # Zero Gradients
    # Manually Zero Grad
    #w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w,b] = model.parameters()
        # print(f'epoch {epoch + 1} : w = {w:.3f}, loss = {l:.8f}')
        print(f'epoch {epoch + 1} : w = {w[0][0].item():.3f}, loss = {l:.8f}')

#print(f'Prediction After Training: f(5) = {forward(5):.3f}')  
print(f'Prediction After Training: f(5) = {model(X_test).item():.3f}')


