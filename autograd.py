import torch
x = torch.randn(3,requires_grad=True)
print(x)
y = x + 2
print(y)

z = y * y * 2
print(z)
z = z.mean()
print(z)
z.backward() #dz/dx 
# Note z should be a scaler value, if it is not then we need to multiply it with scaler vector
# eg. v = torch.tensor([0.1,1.0,0.001], dtype = torch.float32)
# z.backward(v)
print(x.grad)

first = torch.randn(3,requires_grad=True)
print(first)

first.requires_grad_(False)
print(first)

second = first.detach()
print(second)

with torch.no_grad():
    y = x + 2
    print(y)
y = x + 2
print(y)

#Training Steps eg way 1

weights = torch.ones(4,requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward() #Calculate the gradient
    print(weights.grad)
    weights.grad.zero_() #Always clear the grad after the backward function else it will update the values of the weights 




