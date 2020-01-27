import re
import os
import torch
import torch.nn as nn

class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def save(self, path, epoch, loss=None, optimizer=None, remove_old=False):
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

    def load(self, path, loadChkpt, mapLocation='cpu'):
        regex = re.compile(r'^chkpt_(\d*).pt$')
        chkpt = None
        for file in os.listdir(path):
            result = regex.match(file)
            if result:
                if chkpt is None or result.group(1) > chkpt:
                    chkpt = result.group(1)

        file = os.path.join(path, f"chkpt_{chkpt}.pt")
        checkpoint = torch.load(file, mapLocation=mapLocation)

        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch'],\
               checkpoint['optimizer_state_dict'],\
               checkpoint['loss']
