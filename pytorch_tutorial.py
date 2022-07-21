import torch
import numpy as np
x = torch.empty(2,2,2,3,dtype=torch.float16)
print(x)
y = torch.ones(2,2,dtype=torch.int)
print(y)
print(y.size())
z = torch.tensor([2.5,0.1])
print(z)
# trailing _ ==> inplace = True

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# Note if we are on CPU then both a and b will share the same space
# If we change b, a will also be changed
a.add_(1)
print(a)
print(b)

#
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5,device=device)
    y = torch.ones(5)
    y = y.to(device)
    #This will run on Cuda
    z = x + y
    # z.numpy() This line will throw error as numpy can only handle CPU Tensor
    z = z.to("cpu") # move back to cpu 

x = torch.ones(5,requires_grad=True) # by default it is False
# Whenever you have a variable you need to optimize later, then you need gradient
# so put requires_grad=True
print(x)