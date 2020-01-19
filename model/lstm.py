import torch
import torch.nn as nn
from model.modelBase import ModelBase

class Classifier(ModelBase):

    def __init__(self, inputSize, outputSize, hiddenSize, numLayers=1,
                 dropout=0.):
        super().__init__()

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

        returns         Dict with features packed into a
                        torch.utils.rnn.PackedSequence and labels converted to
                        torch.LongTensor
    """
    batchFeatures = []
    batchLabels   = []

    for sample in batchSequence:
        batchFeatures.append(sample['features'])
        batchLabels.append(sample['label'])

    return {'features': torch.nn.utils.rnn.pack_sequence(batchFeatures,
                        enforce_sorted=False),
            'label': torch.LongTensor(batchLabels)}
