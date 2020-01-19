import os
import re
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):

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
