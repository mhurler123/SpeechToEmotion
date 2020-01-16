import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


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

def loadData(filename, valPerc=0):
    data = LogMelDataset(filename)
    splitIndex = int(len(data)*(1-valPerc))
    trainSet, valSet = torch.utils.data.random_split(data,[splitIndex,len(data)-splitIndex])
    return data, trainSet, valSet

def collateRNN(batchSequence):
    batchFeatures = []
    batchLabels   = []

    for sample in batchSequence:
        batchFeatures.append(sample['features'])
        batchLabels.append(sample['label'])

    return {'features': torch.nn.utils.rnn.pack_sequence(batchFeatures,
                        enforce_sorted=False),
            'label': torch.LongTensor(batchLabels)}

def collateCNN(batchSequence):
    batchFeatures = []
    batchLabels   = []

    maxSeqLen = 0
    for sample in batchSequence:
        seqLen = sample['features'].size()[0]
        if seqLen > maxSeqLen:
            maxSeqLen = seqLen
    maxSeqLen=2000

    for sample in batchSequence:
        feature = torch.FloatTensor(sample['features'])
        pad = (0, 0, 0, maxSeqLen - feature.shape[0])
        pdZeroFeature = [F.pad(feature, pad, 'constant', 0).tolist()]
        batchFeatures.append(pdZeroFeature)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}

def testBatchingRNN(filename):
    # test batching
    trainset = LogMelDataset(filename)

    dataloader = DataLoader(trainset, batch_size=10, shuffle=False,
                            num_workers=4, collate_fn=collateRNN)
    for numBatch, batch in enumerate(dataloader):
        inputs = batch['features']
        labels = batch['label']
        print(numBatch, len(inputs), len(labels))
        if (numBatch == 0):
            print(inputs)

def testBatchingCNN(filename):
    # test batching
    trainset = LogMelDataset(filename)

    dataloader = DataLoader(trainset, batch_size=10, shuffle=False,
                            num_workers=4, collate_fn=collateCNN)
    for numBatch, batch in enumerate(dataloader):
        inputs = batch['features']
        labels = batch['label']
        print(numBatch, inputs.size(), labels.size())
        if (numBatch == 0):
            print(inputs)

def printDataset(filename):
    trainset = LogMelDataset(filename)

    for i in range(len(trainset)):
        sample = trainset[i]
        print(i, sample['features'].size(), len(sample['label']))

def printMaxSeqLen(filename):
    trainset = LogMelDataset(filename)

    maxSeqLen = 0
    for i in range(len(trainset)):
        sample = trainset[i]
        seqLen = sample['features'].size()[0]
        if  seqLen > maxSeqLen:
            maxSeqLen = seqLen
    print ("Longest sequence: ", maxSeqLen)

#testBatchingRNN('train.json')
#testBatchingCNN('train.json')
#printDataset('train.json')
#printMaxSeqLen('train.json')
