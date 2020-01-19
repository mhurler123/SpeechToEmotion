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

class Classifier(nn.Module):
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


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier.

        batchSequence   List of data to be batched

        returns         Dict with features which are zero padded to match the
                        maximum sequence length and labels converted to
                        torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    maxSeqLen=2000

    for sample in batchSequence:
        feature = torch.FloatTensor(sample['features'])
        pad = (0, 0, 0, maxSeqLen - feature.shape[0])
        pdZeroFeature = [F.pad(feature, pad, 'constant', 0).tolist()]
        batchFeatures.append(pdZeroFeature)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}
