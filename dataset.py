import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LogMelDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw = self.data[str(idx)]
        return {'features': torch.FloatTensor(raw['features']),
                'label': onehot(raw['valence'], raw['activation'])}

    def load(self):
        with open(self.filename) as jsonFile:
            self.data = json.load(jsonFile)

    def numFeaturesPerFrame(self):
        # each frame in the logMel sequence should contain 26 features
        return len(self.data['0']['features'][0])

    def labelSize(self):
        return 4 # one class for each quarter in the (valence, activation) space

def onehot(valence, activation):
    lookup = {(0,0): [1,0,0,0],
              (1,0): [0,1,0,0],
              (0,1): [0,0,1,0],
              (1,1): [0,0,0,1]}
    return lookup[(valence, activation)]

def loadAndSplitData(filename, valPerc=0):
    """
    Loads dataset and splits it into training and validation.
    """
    data = LogMelDataset(filename)
    splitIndex = int(len(data)*(1-valPerc))
    trainSet, valSet = torch.utils.data.random_split(data,[splitIndex,len(data)-splitIndex])
    return data, trainSet, valSet
