# Code to train Convolution Neural Network

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
num_epochs = 150
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
# CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(576, 256),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(p=0.5),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

CNN_model = CNN().to(device)
print("The model is: ")
print(CNN_model)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN_model.parameters(), lr=learning_rate)

print_acc = True
loss_epoch_arr = []
for epoch in tqdm(range(num_epochs)):
    for train_batch_idx, batch in enumerate(tqdm(train_data_loader, desc='Train Iterator', leave=False)):
        #keeping the network in training mode     
        CNN_model.train()
        
        inputs, labels = batch     
        #moving the input and labels to gpu     
        inputs, labels = inputs.to(device), labels.to(device)     
        #clear the gradients     
        optimizer.zero_grad()     
        #forward pass     
        outputs = CNN_model(inputs)   
        loss = loss_func(outputs, labels)     
        #backward pass     
        loss.backward()     
        optimizer.step()     
    loss_epoch_arr.append(loss.item())
    if(print_acc and (epoch+1)%15 == 0):
        train_acc = evaluation(CNN_model, train_data_loader, device, get_cm=False)        
        test_acc  = evaluation(CNN_model, test_data_loader, device, get_cm=False)
        print("Epoch: %d, train_loss: %.5f, train_acc: %.2f%%, test_acc: %.2f%%" % (
                        epoch+1, loss_epoch_arr[-1], train_acc, test_acc))


#plotting the loss chart 
plt.plot(loss_epoch_arr)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("CNN_train_loss.png")
plt.show()

cm_train, train_acc = evaluation(CNN_model, train_data_loader, device, get_cm=True)
cm_test, test_acc= evaluation(CNN_model, test_data_loader, device, get_cm=True)
plot_confusion_matrix(cm_train, normalize=False, target_names=[label_name[i] for i in range(10)], title="CNN_Confusion Matrix - Train")
plot_confusion_matrix(cm_test, normalize=False, target_names=[label_name[i] for i in range(10)], title="CNN_Confusion Matrix - Test")
print('Train acc: %0.2f, Test acc: %0.2f' % (train_acc, test_acc))

# Save the model
torch.save(CNN_model.state_dict(), './model/cnn')