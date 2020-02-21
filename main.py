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

print(">> Name: Rishabh Gupta")
print(">> Dept: CSA")
print(">> Sr N: 15960")

# Hyper parameters
batch_size = 4096
data_loader_workers = 10

label_name = {0 : 'T-Shirt', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt',
              7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'};

# Define a transform to normalize the data
transform_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
# Loading the test data
test_data = FashionMNIST(root="./data", train=False, download=True, transform=transform_test)

# Data Loaders
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# Define the models

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



MLP_model = MLP().to(device)
MLP_model.load_state_dict(torch.load('./model/mlp'))
CNN_model = CNN().to(device)
CNN_model.load_state_dict(torch.load('./model/cnn'))

# Evaluation
cm_mlp, acc_mlp, predicted_mlp, target_mlp = evaluation(MLP_model, test_data_loader, device, get_cm=True, get_outputs=True)
plot_confusion_matrix(cm_mlp, normalize=False, target_names=[label_name[i] for i in range(10)], title="MLP_Confusion Matrix - Test")
cm_cnn, acc_cnn, predicted_cnn, target_cnn = evaluation(CNN_model, test_data_loader, device, get_cm=True, get_outputs=True)
plot_confusion_matrix(cm_cnn, normalize=False, target_names=[label_name[i] for i in range(10)], title="CNN_Confusion Matrix - Test")

print('MLP Test acc: %0.2f, CNN Test acc: %0.2f' % (acc_mlp, acc_cnn))

with open("multi-layer-net.txt", 'w') as f:
    f.write("Accuracy on Test Data : " + str(acc_mlp) + "\n")
    f.write("gt_label,pred_label\n")
    for i in range(len(predicted_mlp)):
        f.write(str(int(target_mlp[i]))+","+str(int(predicted_mlp[i]))+"\n")

with open("convolution-neural-net.txt", 'w') as f:
    f.write("Accuracy on Test Data : " + str(acc_cnn) + "\n")
    f.write("gt_label,pred_label\n")
    for i in range(len(predicted_cnn)):
        f.write(str(int(target_cnn[i]))+","+str(int(predicted_cnn[i]))+"\n")

        