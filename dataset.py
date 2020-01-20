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
        if 'valence' in raw:
            return {'features': torch.FloatTensor(raw['features']),
                    'label': onehot(raw['valence'], raw['activation'])}
        else:
            return {'features': torch.FloatTensor(raw['features']),
                    'label': onehot(0, 0)}

    def load(self):
        with open(self.filename) as jsonFile:
            self.data = json.load(jsonFile)

    def numFeaturesPerFrame(self):
        # each frame in the logMel sequence should contain 26 features
        return len(self.data['0']['features'][0])

    def labelSize(self):
        return 4 # one class for each quadrant in the (valence, activation) space

def onehot(valence, activation):
    lookup = {(0,0): [1,0,0,0],
              (1,0): [0,1,0,0],
              (0,1): [0,0,1,0],
              (1,1): [0,0,0,1]}
    return lookup[(valence, activation)]

def onehotRev(classId):
    lookup = [(0,0),
              (1,0),
              (0,1),
              (1,1)]
    return lookup[classId]

def loadAndSplitData(filename, valPerc=0):
    """
    Loads dataset and splits it into training and validation.

        filename   Filename of json dataset

        valPerc    Amount of validation data

        returns    List comprising the whole, training and validation dataset
    """
    data = LogMelDataset(filename)
    splitIndex = int(len(data)*(1-valPerc))
    trainSet, valSet = torch.utils.data.random_split(data,[splitIndex,len(data)-splitIndex])
    return trainSet, valSet, data
