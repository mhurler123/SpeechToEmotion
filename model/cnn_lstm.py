import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modelBase import ModelBase

FRAME_SIZE  = 50
NUM_FRAMES  = 20
MAX_SEQ_LEN = FRAME_SIZE * NUM_FRAMES


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
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(3,3),stride=(1,1), padding_mode='same'),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,2)),
            nn.Dropout2d(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3,3),stride=(1,1), padding_mode='same'),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3,3), stride=(1,1), padding_mode='same'),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25)
        )

        # A linear layer maps the high-level features to an emotion prediction
        self.layer4 = nn.Sequential (
            nn.Linear(in_features=40, out_features=4),
            nn.ReLU()
        )

        # The lstm network turns the emotion predictions for each frame into a
        # prediction for the whole sequence
        self.lstm = nn.LSTM(input_size=4, hidden_size=4,
                            num_layers=1, dropout=0)

    def cnn(self, inputs):
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        # outputs.shape is (batch_size * timesteps, 10, H, W)
        # flatten to shape (batch_size * timesteps, 10*H*W)
        outputs = outputs.view(outputs.size(0), -1) # flatten
        outputs = self.layer4(outputs)
        return outputs

    def forward(self, inputs):
        batchSize, numFrames, numChannels, frameSize, numFeatures = inputs.size()
        cnnIn   = inputs.view(batchSize * timesteps, numChannels, frameSize,
                              numFeatures)
        cnnOut  = self.cnn(cnnIn)
        lstmIn  = cnnOut.view(batchSize, numFrames, -1)
        outputs, (hn, cn) = self.lstm(lstmIn) # by default (h0, c0) are zero
        outputs = hn[0] # we will only use the last hidden state
        return outputs


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier.

        batchSequence   List of data to be batched

        returns         Dict with features of shape
                        (batchSize, NUM_FRAMES, 1, FRAME_SIZE, 26)
                        and labels converted to torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    for sample in batchSequence:
        # pad all sequences to the same length
        feature = torch.FloatTensor(sample['features'])
        pad = (0, 0, 0, MAX_SEQ_LEN - feature.shape[0])
        pdZeroFeature = F.pad(feature, pad, 'constant', 0).tolist()

        # subdivide the temporal sequence into frames
        frames = []
        for frameId in range(NUM_FRAMES):
            start = frameId * FRAME_SIZE
            end   = start + FRAME_SIZE
            frame = pdZeroFeature[start:end]
            # add additional dimension for channel (needed for cnn)
            frames.append([frame])

        batchFeatures.append(frames)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}
