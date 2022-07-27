'''
The softmax, or “soft max,” mathematical function can be thought 
to be a probabilistic or “softer” version of the argmax function.
Softmax : Output will be squashed between 0 and 1(Probabilities)

'''


import torch
import torch.nn as nn
import numpy as np

# We take exponential so that values is always positive
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)


x = torch.tensor([2.0,1.0,0.1])
outputs = torch.softmax(x,dim=0)
print('softmax tensor:', outputs)


'''
Cross Entropy Loss usually works with Softmax.
Lower the loss is the better the prediction is
e.g. Y = [1,0,0] Y' = [0.7,0.2,0.1] Loss will be 0.35(lower) 
Y = [1,0,0] Y' = [0.1,0.3,0.6] Loss will be 2.35(larger) 
We calculate Y' using softmax
'''

def cross_entropy(actual,predicted):
    loss = - np.sum(actual  * np.log(predicted))
    return loss # / float(predicted.shape[0]) Number of samples


# y must be one hot encoded
# if class 0 : [1,0,0]
# if class 1 : [0,1,0]
# if class 2 : [0,0,1]

Y = np.array([1,0,0])

Y_pred_good = np.array([0.7,0.2,0.1]) # Applied Softmax
Y_pred_bad = np.array([0.1,0.3,0.6]) # Applied Softmax
l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')



# Pytorch

# Note nn.CrossEntropyLoss applies nn.LogSoftmax and nn.NLLLoss (negative log likelihood loss)
# So no softmax should be applied in last layer in NN
# Y has class labels, not One-Hot
# Y_pred has raw scores (logits), no Softmax!

loss = nn.CrossEntropyLoss()
Y = torch.tensor([0]) # Y = torch.tensor([1])  Y = torch.tensor([2])
# nsamples * nclasses = 1*3
Y_pred_good = torch.tensor([[2.0,1.0,0.1]]) #Raw value didnt apply softmax
Y_pred_bad = torch.tensor([[0.5,2.0,0.3]])  #Raw value didnt apply softmax

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())

_,predictions1 = torch.max(Y_pred_good,1)
_,predictions2 = torch.max(Y_pred_bad,1) 
# This will choose the highest probability
print(predictions1)
print(predictions2)


# 3 samples
Y = torch.tensor([2,0,1]) # Y = torch.tensor([1])  Y = torch.tensor([2])
# nsamples * nclasses = 3*3
Y_pred_good = torch.tensor([[0.1,1.0,2.1], [2.0,1.0,0.1],[0.1,3.0,0.1]]) #Raw value didnt apply softmax
Y_pred_bad = torch.tensor([[2.1,1.0,0.1], [0.1,1.0,2.1],[0.1,3.0,0.1]])  #Raw value didnt apply softmax

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())

_,predictions1 = torch.max(Y_pred_good,1)
_,predictions2 = torch.max(Y_pred_bad,1) 
# This will choose the highest probability
print(predictions1)
print(predictions2)
