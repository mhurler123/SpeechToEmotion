import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modelBase import ModelBase

TEMPORAL_FRAME_SIZE = 50
NUM_FRAMES          = 20
MAX_SEQ_LEN         = TEMPORAL_FRAME_SIZE * NUM_FRAMES

class Classifier(ModelBase):
    def __init__(self, inputSize, outputSize, hiddenSize=26, numLayers=1,
                 dropout=0.):
        super().__init__()

        # Start off with a cnn to extract some high-level features
        

        # The LSTM Network
        self.lstm = nn.LSTM(input_size=inputSize, hidden_size=hiddenSize,
                            num_layers=numLayers, dropout=dropout)

        # A single linear unit (fully connected layer) will convert
        # the output of the LSTM with size hiddenSize to the desired output size
        self.fc = nn.Linear(in_features=hiddenSize, out_features=outputSize)

    def forward(self, inputs):
        outputs, (hn, cn) = self.lstm(inputs) # by default (h0, c0) are zero
        outputs = self.fc(hn[0])
        return outputs


def collate(batchSequence):
    """
    Since the data can contain sequences of different size, the default batching
    of torch.utils.data.DataLoader can not be used.
    This tells the DataLoader how a sequence of raw data must be batched in
    order to work with the classifier.

        batchSequence   List of data to be batched

        returns         Dict with features of shape
                        (batchSize, NUM_FRAMES, TEMPORAL_FRAME_SIZE, 26, 1)
                        and labels converted to torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    # pad all feature sequences to the same length
    for sample in batchSequence:
        feature = torch.FloatTensor(sample['features'])
        pad = (0, 0, 0, MAX_SEQ_LEN - feature.shape[0])
        pdZeroFeature = [F.pad(feature, pad, 'constant', 0).tolist()]

        temporalFrames = []
        for frameId in range(NUM_FRAMES)
            start = frameId * TEMPORAL_FRAME_SIZE
            end   = start + TEMPORAL_FRAME_SIZE
            frame = pdZeroFeature[start:end]
            # add additional dimension for channel (needed for cnn)
            frame = [ [x] for x in frame]
            temporalFrames.append(frame)

        batchFeatures.append(temporalFrames)
        batchLabels.append(sample['label'])

    return {'features': torch.FloatTensor(batchFeatures),
            'label': torch.LongTensor(batchLabels)}
