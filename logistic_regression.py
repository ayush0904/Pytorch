# 1) Design model (input, output size, forward pass)
# 2) Construct Loss and Optimizer
# 3) Training Loop
#     - Forward Pass : Compute Prediction
#     - Backward Pass : Gradients
#     - Update Weights

from os import access
from statistics import mode
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features )

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

#Scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

#model
#f = wx + b, sigmoid at the end

class LogisticRegression(nn.Module):

    def __init__(self,n_input_features) -> None:
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted
    

model = LogisticRegression(n_features)

# Loss
learning_rate = 0.01
criterian = nn.BCELoss()
optimizer  = torch.optim.SGD(model.parameters(), lr = learning_rate)

# training loop

num_epocs = 100
for epoch in range(num_epocs):

    # forward pass
    y_predicted = model(X_train)
    loss = criterian(y_predicted,y_train)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    #zero gradient
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch : {epoch + 1}, loss : {loss.item(): .4f}')

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc: .4f}')