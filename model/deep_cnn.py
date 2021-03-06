import torch
import torch.nn as nn
import numpy as np
from model.modelBase import ModelBase

FEATURE_SIZE         = 26
LABEL_SIZE           = 4
MAX_SEQ_LEN          = 1000
CNN1_CHANNELS        = 128
CNN2_CHANNELS        = 128
CNN3_CHANNELS        = 128
CNN4_CHANNELS        = 64
CNN5_CHANNELS        = 64
CNN6_CHANNELS        = 64
CNN1_MAX_POOL_KERNEL = (2,1)
CNN2_MAX_POOL_KERNEL = (2,1)
CNN3_MAX_POOL_KERNEL = (2,1)
CNN4_MAX_POOL_KERNEL = (2,1)
CNN5_MAX_POOL_KERNEL = (2,1)
CNN6_MAX_POOL_KERNEL = (2,2)

def getLinearLayerInputSize():
    scaleSeq = CNN1_MAX_POOL_KERNEL[0]*CNN2_MAX_POOL_KERNEL[0]*\
            CNN3_MAX_POOL_KERNEL[0]*CNN4_MAX_POOL_KERNEL[0]*\
            CNN5_MAX_POOL_KERNEL[0]*CNN6_MAX_POOL_KERNEL[0]
    scaleFeature = CNN1_MAX_POOL_KERNEL[1]*CNN2_MAX_POOL_KERNEL[1]*\
            CNN3_MAX_POOL_KERNEL[1]*CNN4_MAX_POOL_KERNEL[1]*\
            CNN5_MAX_POOL_KERNEL[1]*CNN6_MAX_POOL_KERNEL[1]
    return CNN6_CHANNELS * int(MAX_SEQ_LEN/scaleSeq)\
            * int(FEATURE_SIZE/scaleFeature)

LINEAR1_IN = getLinearLayerInputSize()
LINEAR2_IN = 100
LINEAR3_IN = 100

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
            nn.MaxPool2d(kernel_size=CNN1_MAX_POOL_KERNEL, stride=(2,1)),
            nn.Dropout2d(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=CNN1_CHANNELS, out_channels=CNN2_CHANNELS,
                    kernel_size=(3,3),stride=(1,1), padding = 1,
                    padding_mode='same'),
            nn.BatchNorm2d(CNN2_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN2_MAX_POOL_KERNEL, stride=(2, 1)),
            nn.Dropout2d(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=CNN2_CHANNELS, out_channels=CNN3_CHANNELS,
                    kernel_size=(3,3),stride=(1,1), padding = 1,
                    padding_mode='same'),
            nn.BatchNorm2d(CNN3_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN3_MAX_POOL_KERNEL, stride=(2, 1)),
            nn.Dropout2d(0.25)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=CNN3_CHANNELS, out_channels=CNN4_CHANNELS,
                    kernel_size=(3,3),stride=(1,1), padding = 1,
                    padding_mode='same'),
            nn.BatchNorm2d(CNN4_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN4_MAX_POOL_KERNEL, stride=(2, 1)),
            nn.Dropout2d(0.25)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=CNN4_CHANNELS, out_channels=CNN5_CHANNELS,
                    kernel_size=(3,3),stride=(1,1), padding = 1,
                    padding_mode='same'),
            nn.BatchNorm2d(CNN5_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN5_MAX_POOL_KERNEL, stride=(2, 1)),
            nn.Dropout2d(0.25)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=CNN5_CHANNELS, out_channels=CNN6_CHANNELS,
                    kernel_size=(3,3),stride=(1,1), padding = 1,
                    padding_mode='same'),
            nn.BatchNorm2d(CNN6_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN6_MAX_POOL_KERNEL, stride=(2, 2)),
            nn.Dropout2d(0.25)
        )

        self.layer7 =  nn.Sequential(
            nn.Linear(LINEAR1_IN, LINEAR2_IN),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer8 = nn.Sequential(
            nn.Linear(LINEAR2_IN, LINEAR3_IN),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.layer9 =  nn.Sequential(
            nn.Linear(LINEAR3_IN, LABEL_SIZE),
            nn.Dropout(0.2)
        )

    def forward(self, input_seq):
        out = self.layer1(input_seq)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1) # flatten
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        return out


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier.

        batchSequence   List of data to be batched

        returns         Dict with features that are cropped or padded by
                        repeating the feature sequence to match desired maximum
                        sequence length and labels converted to torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    for sample in batchSequence:
        feature = sample['features'].tolist()
        seqLen = len(feature)

        # perform padding
        padLen = MAX_SEQ_LEN - seqLen
        if padLen > 0:
            feature = np.pad(feature, ((0, padLen), (0, 0)),
                'wrap').tolist()
        else:
            feature = feature[:MAX_SEQ_LEN:]

        # add additional dimension for channel
        feature = [feature]

        # append cropped or padded data to batch
        batchFeatures.append(feature)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label'   : torch.LongTensor(batchLabels)}
