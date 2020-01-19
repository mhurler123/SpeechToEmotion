#%%
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
#%%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNNClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self):
        super(CNNClassifier, self).__init__()
        num_classes = 4

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128 , kernel_size=(3,3),stride=(1,1), padding_mode='same'),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=(2,2)),
                nn.Dropout2d(0.5)
                )

        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3),stride=(1,1), padding_mode='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.25)
                )

        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3),stride=(1,1), padding_mode='same'),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.25)
                )

        # conv2d with kernel 3 and out 13 will crop to 12x12
        # max pool 12/4= 3
        # therefore 3*3*13
        self.fc = nn.Linear(15872,num_classes)

    def forward(self, input_seq):
        out = self.layer1(input_seq)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # flatten
        print(out.shape)
        out = self.fc(out)
        return out
