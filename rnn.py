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

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper paramters
# We will treat one dimension of image as sequence and another as input/feature size 
input_size = 28 
sequence_length = 28
hidden_size = 128 # try different size 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001
num_layers = 2 # stacking 2 RNN together, second RNN takes the output of the first RNN 

#MNIST
train_dataset = torchvision.datasets.MNIST(root='./data' , train = True, transform = transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root='./data' , train = False, transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle= True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle= False)

examples = iter(train_loader)

samples,labels = examples.next()

print(samples.shape, labels.shape)
#torch.Size([100, 1, 28, 28]) torch.Size([100]) --> 100 = batch_size , 1 = since we have black and white image, 
# 28*28 = size of image .. labels = 100

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')
#plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, num_classes):
        super(RNN, self).__init__()
       
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first= True)
        #self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first= True)
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first= True)
        # Input must have the shape (x) -> (batch_size , seq , input_size) if batch_first = True

        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0) , self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device) # FOR LSTM
        out , _ = self.rnn(x,h0)
        #out , _ = self.gru(x,h0)
        #out , _ = self.lstm(x,(h0,c0)) 
        # out : batch_size, seq_length, hidden_size
        # out (N,28,128)
        out = out[:, -1, : ] # We need the last hidden state
        # out (N,128)
        out = self.fc(out)
        return out


        

model = RNN(input_size,hidden_size,num_layers,num_classes)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # we need to reshape our images
        # Present Shape = [100,1,28,28]
        # Required Shape = [100,28,28]
        images = images.reshape(-1,sequence_length,input_size).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1,sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs  = model(images)

        #value, index
        _, predictions =  torch.max(outputs,1)

        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
        
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
