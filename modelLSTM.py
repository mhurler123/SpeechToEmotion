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


    def save(self, path, epoch, loss=None, optimizer=None, remove_old=True):
        print("Saving model... ", end="")
        file = os.path.join(path, f"chkpt_{epoch}.pt")

        if remove_old:
            regex = re.compile(r'^chkpt_(\d*).pt$')
            for _file in os.listdir(path):
                result = regex.match(_file)
                if result:
                    if int(result.group(1)) < epoch:
                        try:
                            os.remove(_file)
                        except FileNotFoundError:
                            pass

        chkpt = {
            'epoch': epoch,
            'model_state_dict': self.state_dict()
            }
        if optimizer is not None:
            chkpt.update(optimizer_state_dict=optimizer.state_dict())
        if loss is not None:
            chkpt.update(loss=loss)

        torch.save(chkpt, file)


    def load(self, path, load_chkpt):
        regex = re.compile(r'^chkpt_(\d*).pt$')
        chkpt = None
        for file in os.listdir(path):
            result = regex.match(file)
            if result:
                if chkpt is None or result.group(1) > chkpt:
                    chkpt = result.group(1)

        file = os.path.join(path, f"chkpt_{chkpt}.pt")
        checkpoint = torch.load(file)

        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'],\
               checkpoint['optimizer_state_dict'],\
               checkpoint['loss']
