#%%
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modelBase import ModelBase

FEATURE_SIZE         = 26
LABEL_SIZE           = 4
MAX_SEQ_LEN          = 1000
CNN1_CHANNELS        = 128
CNN2_CHANNELS        = 64
CNN3_CHANNELS        = 64
CNN1_MAX_POOL_KERNEL = (2,2)
CNN2_MAX_POOL_KERNEL = (2,2)
CNN3_MAX_POOL_KERNEL = (2,2)

def getLinearLayerInputSize():
    scaleSeq = CNN1_MAX_POOL_KERNEL[0]*CNN2_MAX_POOL_KERNEL[0]*\
            CNN3_MAX_POOL_KERNEL[0]
    scaleFeature = CNN1_MAX_POOL_KERNEL[1]*CNN2_MAX_POOL_KERNEL[1]*\
            CNN3_MAX_POOL_KERNEL[1]
    return CNN3_CHANNELS * int(MAX_SEQ_LEN/scaleSeq)\
            * int(FEATURE_SIZE/scaleFeature)

LINEAR_IN = getLinearLayerInputSize()

class Classifier(ModelBase):
    """docstring for LSTMClassifier"""
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=CNN1_CHANNELS ,
                        kernel_size=(3,3),stride=(1,1), padding = 1,
                        padding_mode='same'),
                nn.BatchNorm2d(CNN1_CHANNELS),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=CNN1_MAX_POOL_KERNEL, stride=(2,2)),
                nn.Dropout2d(0.5)
                )

        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=CNN1_CHANNELS, out_channels=CNN2_CHANNELS,
                        kernel_size=(3,3),stride=(1,1), padding = 1,
                        padding_mode='same'),
                nn.BatchNorm2d(CNN2_CHANNELS),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=CNN2_MAX_POOL_KERNEL, stride=(2, 2)),
                nn.Dropout2d(0.25)
                )

        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=CNN2_CHANNELS, out_channels=CNN3_CHANNELS,
                        kernel_size=(3,3),stride=(1,1), padding = 1,
                        padding_mode='same'),
                nn.BatchNorm2d(CNN3_CHANNELS),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=CNN3_MAX_POOL_KERNEL, stride=(2, 2)),
                nn.Dropout2d(0.25)
                )

        # conv2d with kernel 3 and out 13 will crop to 12x12
        # max pool 12/4= 3
        # therefore 3*3*13
        self.fc =  nn.Sequential(
            nn.Linear(LINEAR_IN, LABEL_SIZE),
            nn.ReLU(),
            nn.Dropout2d(0.25)
        )

    def forward(self, input_seq):
        out = self.layer1(input_seq)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.fc(out)
        return out


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier.

        batchSequence   List of data to be batched

        returns         Dict with features that are reflection padded to match
                        the maximum sequence length and labels converted to
                        torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    maxSeqLen=1000

    for sample in batchSequence:
        feature = torch.FloatTensor(sample['features'])
        feature = feature.reshape(1, 1, feature.shape[0], feature.shape[1])
        padLen = MAX_SEQ_LEN - seqLen

        # workaround if multiple reflections are necessary
        repetitions = padLen / seqLen
        if repetitions >= 1.:
            for i in range(int(repetitions)):
                reflectionPad = nn.ReflectionPad2d((0, 0, 0, seqLen-1))
                feature = reflectionPad(feature)
                reflectionPad = nn.ReflectionPad2d((0, 0, 0, 1))
                feature = reflectionPad(feature)
            reflectionPad = nn.ReflectionPad2d((0, 0, 0, padLen%seqLen))
            feature = reflectionPad(feature)
        else:
            pad = (0, 0, 0, padLen)
            reflectionPad = nn.ReflectionPad2d(pad)
            feature = reflectionPad(feature)
        feature = feature[0][0].tolist()

        batchFeatures.append(pdZeroFeature)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}
