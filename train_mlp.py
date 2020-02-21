# Code to train MultiLayer Perceptron

import numpy as np
import os

from tqdm import tqdm
#from tqdm.autonotebook import tqdm

from matplotlib import pyplot as plt
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

# Utils
from utils import plot_confusion_matrix
from utils import evaluation

# Hyper parameters
num_epochs = 200
batch_size = 4096
learning_rate = 0.001
data_loader_workers = 10

label_name = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};

# Define a transform to normalize the data
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
# Loading the train and test data
train_data = FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
test_data = FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

# Data Loaders
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# Define the model
# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 2048),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.clf = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1))
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.clf(out)
        return out

MLP_model = MLP().to(device)
print("The model is: ")
print(MLP_model)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(MLP_model.parameters(), lr=learning_rate)

print_acc = True
loss_epoch_arr = []
for epoch in tqdm(range(num_epochs)):
    for train_batch_idx, batch in enumerate(tqdm(train_data_loader, desc='Train Iterator', leave=False)):
        #keeping the network in training mode     
        MLP_model.train()
        
        inputs, labels = batch     
        #moving the input and labels to gpu     
        inputs, labels = inputs.to(device), labels.to(device)     
        #clear the gradients     
        optimizer.zero_grad()     
        #forward pass       
        outputs = MLP_model(inputs) 
        loss = loss_func(outputs, labels)     
        #backward pass     
        loss.backward()     
        optimizer.step()     
    loss_epoch_arr.append(loss.item())
    if(print_acc and (epoch+1)%20 == 0):
        train_acc = evaluation(MLP_model, train_data_loader, device, get_cm=False)
        test_acc  = evaluation(MLP_model, test_data_loader, device, get_cm=False)
        print("Epoch: %d, train_loss: %.5f, train_acc: %.2f%%, test_acc: %.2f%%" % (
                        epoch+1, loss_epoch_arr[-1], train_acc, test_acc))


#plotting the loss chart 
plt.plot(loss_epoch_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("MLP_train_loss.png")
plt.show()

cm_train, train_acc = evaluation(MLP_model, train_data_loader, device, get_cm=True)
cm_test, test_acc= evaluation(MLP_model, test_data_loader, device, get_cm=True)
plot_confusion_matrix(cm_train, normalize=False, target_names=[label_name[i] for i in range(10)], title="MLP_Confusion Matrix - Train")
plot_confusion_matrix(cm_test, normalize=False, target_names=[label_name[i] for i in range(10)], title="MLP_Confusion Matrix - Test")
print('Train acc: %0.2f, Test acc: %0.2f' % (train_acc, test_acc))

# Save the model
torch.save(MLP_model.state_dict(), './model/mlp')