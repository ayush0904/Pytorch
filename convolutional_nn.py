from cProfile import label
import nntplib
from pickletools import optimize
from random import sample
from turtle import forward
import datasets
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper paramters
# input_size = 784 #28*28
# hidden_size = 500
# num_classes = 10
num_epochs = 4
batch_size = 4
learning_rate = 0.001


#dataset has PILImage images from rang [0,1]
# We transform them to Tensors of normalized range [-1,1]


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#CIFAR10
train_dataset = torchvision.datasets.CIFAR10(root='./data' , train = True, transform = transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data' , train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle= True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle= False)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')




class ConvNet(nn.Module):
    def __init__(self):

        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)

        self.conv2 = nn.Conv2d(6,16,5)
        # After this layer the output size will be [4,16,5,5] 
        # That is why we are put 16*5*5
        self.fc1 = nn.Linear(16*5*5, 120) # 120 can be changed , 16*5*5 must be fixed
        self.fc2 = nn.Linear(120,84) # 84 can be changed
        self.fc3 = nn.Linear(84,10) # 10 should be same since there are 10 output classes

       

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5) #Flatten our tensor befor applying fully connected layer
        x =  F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x =  F.relu(self.fc3(x))
        return x



       

model = ConvNet().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss() # for multiclass classification problems
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # we need to reshape our images
        # Origin Shape = 4,3,32,32
        # Required Shape = 4,3,1024
        #  input_channels = 3 output channels = 6 kernal size = 5
       
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs  = model(images)

        #value, index
        _, predictions =  torch.max(outputs,1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
        
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network = {acc}')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]} : {acc} %')
