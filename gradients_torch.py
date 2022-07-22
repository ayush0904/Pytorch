# Scratch
import torch

# f = w * x

# f = 2 * x

X = torch.tensor([1,2,3,4], dtype=torch.float32) 
Y = torch.tensor([2,4,6,8], dtype=torch.float32) 

w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

# model prediction

def forward(x):
    return w * x

#Loss = MSE (Mean Square Error)

def loss(y,y_predicted):
    return ((y_predicted-y)**2).mean()



print(f'Prediction Before Training: f(5) = {forward(5):.3f}')

# Training 
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)

    # Loss 
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward() # dl/dw



    # update weights
    # this operation should not be a part of gradiwnt tracking (computational part)
    with torch.no_grad():
        w -= learning_rate * w.grad

    # Zero Gradients
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f'epoch {epoch + 1} : w = {w:.3f}, loss = {l:.8f}')


print(f'Prediction After Training: f(5) = {forward(5):.3f}')


