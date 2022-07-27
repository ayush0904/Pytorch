'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

import math
import datasets
import numpy as np
import torch 
import torchvision
from torch.utils.data import Dataset,DataLoader

class WineDataset(Dataset):

    def __init__(self, transform=None):
        
        xy = np.loadtxt('./dataset/wine.csv', delimiter = "," , dtype = np.float32, skiprows = 1)
        self.x = xy[:,1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]

        self.transform = transform
        

    def __getitem__(self,index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
        

    def __len__(self):
        return self.n_samples


class ToTensor:

    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
        
class MulTransform:

    def __init__(self,factor):
        self.factor = factor
    
    def __call__(self,sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
#dataset = WineDataset(transform=None)
first_data = dataset[0]
features,labels = first_data
print(features)
print(type(features),type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed) 
first_data = dataset[0]
features,labels = first_data
print(features)
print(type(features),type(labels))
