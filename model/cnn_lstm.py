import torch
import torch.nn as nn
import numpy as np
from model.modelBase import ModelBase

FEATURE_SIZE         = 26
LABEL_SIZE           = 4
FRAME_SIZE           = 20
NUM_FRAMES           = 10
MAX_SEQ_LEN          = FRAME_SIZE * NUM_FRAMES

CNN1_CHANNELS        = 5
CNN2_CHANNELS        = 5
CNN3_CHANNELS        = 2
CNN1_MAX_POOL_KERNEL = (2,1)
CNN2_MAX_POOL_KERNEL = (1,1)
CNN3_MAX_POOL_KERNEL = (1,1)

LSTM_HIDDEN          = 26
LSTM_LAYERS          = 2

def getLinearLayerInputSize():
    scaleFrame = CNN1_MAX_POOL_KERNEL[0]*CNN2_MAX_POOL_KERNEL[0]*\
            CNN3_MAX_POOL_KERNEL[0]
    scaleFeature = CNN1_MAX_POOL_KERNEL[1]*CNN2_MAX_POOL_KERNEL[1]*\
            CNN3_MAX_POOL_KERNEL[1]
    return CNN3_CHANNELS * int(FRAME_SIZE/scaleFrame)\
            * int(FEATURE_SIZE/scaleFeature)

LINEAR_IN            = getLinearLayerInputSize()

class Classifier(ModelBase):
    """
    Idea: Split the logMel sequence of an audio signal e.g. of shape (200, 26)
    into frames of a given size e.g. 10. This results in a sequence of frames
    where each frame contains frame_size logMel features. For the previous
    example this sequence of frames would have the shape (20, 10, 26).
    Now use a cnn to predict the emotion in each frame e.g. map
    from each 10 x 26 frame to one of 4 different classes (one quadrant in the
    valence/arousal) space. In order to get a prediction for the whole audio
    sequence we successively stuff the frame-wise predictions into an lstm.
    """
    def __init__(self):
        super().__init__()

        # Start off with a cnn to extract some high-level features from the seen frame
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=CNN1_CHANNELS,
                kernel_size=(5,5),stride=(1,1), padding=2, padding_mode='same'),
            nn.BatchNorm2d(CNN1_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN1_MAX_POOL_KERNEL,
                stride=CNN1_MAX_POOL_KERNEL),
            nn.Dropout2d(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=CNN1_CHANNELS, out_channels=CNN2_CHANNELS,
                kernel_size=(3,3),stride=(1,1), padding=1, padding_mode='same'),
            nn.BatchNorm2d(CNN2_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN2_MAX_POOL_KERNEL,
                stride=CNN2_MAX_POOL_KERNEL),
            nn.Dropout2d(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=CNN2_CHANNELS, out_channels=CNN3_CHANNELS,
                kernel_size=(3,3), stride=(1,1), padding=1, padding_mode='same'),
            nn.BatchNorm2d(CNN3_CHANNELS),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=CNN3_MAX_POOL_KERNEL,
                stride=CNN3_MAX_POOL_KERNEL),
            nn.Dropout2d(0.25)
        )

        # A linear layer maps the high-level features to an emotion prediction
        self.layer4 = nn.Sequential (
            nn.Linear(in_features=LINEAR_IN, out_features=LSTM_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0)
        )

        # The lstm network turns the emotion predictions for each frame into a
        # prediction for the whole sequence
        self.lstm = nn.LSTM(input_size=LSTM_HIDDEN, hidden_size=LSTM_HIDDEN,
                            num_layers=LSTM_LAYERS, dropout=0)

        self.fc = nn.Sequential (
            nn.Linear(in_features=LSTM_HIDDEN, out_features=LABEL_SIZE),
            nn.ReLU(),
        )

    def cnn(self, inputs):
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        # outputs.shape is (batch_size * timesteps, C, H, W)
        # flatten to shape (batch_size * timesteps, C*H*W)
        outputs = outputs.view(outputs.size(0), -1) # flatten
        outputs = self.layer4(outputs)
        return outputs

    def forward(self, inputs):
        batchSize, numFrames, numChannels, frameSize, numFeatures = inputs.size()
        cnnIn   = inputs.view(batchSize * numFrames, numChannels, frameSize,
                              numFeatures)
        cnnOut  = self.cnn(cnnIn)
        # lstm requires input of shape (timesteps, batch, linear_out)
        lstmIn  = cnnOut.view(batchSize, numFrames, -1).transpose(0,1)
        outputs, (hn, cn) = self.lstm(lstmIn) # by default (h0, c0) are zero
        # only last hidden state of last layer is important to us
        outputs = hn[-1]
        outputs = self.fc(outputs)
        return outputs


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier. Here, each sample sequence is padded by
    repeating the feature sequence and cut into multiple frames.

        batchSequence   List containing the data of a single batch

        returns         Dict with features of shape
                        (batchSize, NUM_FRAMES, 1, FRAME_SIZE, 26) and labels
                        converted to torch.FloatTensor.
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

        # subdivide the temporal sequence into frames
        frames = []
        for frameId in range(NUM_FRAMES):
            start = frameId * FRAME_SIZE
            end   = start + FRAME_SIZE
            frame = feature[start:end]
            # add additional dimension for channel (needed for cnn)
            frames.append([frame])

        batchFeatures.append(frames)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}
