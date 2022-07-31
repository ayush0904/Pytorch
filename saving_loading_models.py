#  Dictionary and Tensor Model
#  It uses torch pikl module 
# torch.save(arg,PATH) 


# @@@@@@@@@

# First is Lazy method

# torch.save(model,PATH)
# model = torch.load(PATH)
# model.eval()

#@@@@@@@@

# @@@@@@@

# Second 

# Save only the parameters STATE DICT
#torch.save(model.state_dict(),PATH)

# model must be created again with parameters

# model = Model(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

#@@@@@@@@

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_input_features) -> None:
        super().__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)

# train your model...

# LAZY

FILE = "model.pth"
#torch.save(model, FILE)

model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)


# Better Way

FILE_BETTER = "model_better.pth"
torch.save(model.state_dict(), FILE_BETTER)

loaded_model = Model(n_input_features=6)
loaded_model.state_dict(torch.load(FILE_BETTER))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)


learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
print(optimizer.state_dict())

# Save a checkpoint

checkpoint = {
    "epoch" : 90,
    "model_state" : model.state_dict(),
    "optim_state" : optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")


loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(),lr=0) #lr is put 0 to test and show that after loading the checkpoint it will take 0.01

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())




# Save your model on GPU and use it on your CPU


# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), PATH)

# device = torch.device('cpu')
# model = Model(*args,**kwargs)
# model.load_state_dict(torch.load(PATH,map_location=device))


# Save your model on GPU and use it on your GPU


# device = torch.device("cuda")
# model.to(device)
# torch.save(model.state_dict(), PATH)

# model = Model(*args,**kwargs)
# model.load_state_dict(torch.load(PATH))
# model.to(device)



# Save your model on CPU and use it on your GPU


# torch.save(model.state_dict(), PATH)
#device = torch.device("cuda")
# model = Model(*args,**kwargs)
# model.load_state_dict(torch.load(PATH,map_location = :"cuda:0"))
# model.to(device)












