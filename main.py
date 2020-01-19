import os
import random
from tqdm import tqdm
import torch
import dataset
from torch.utils.data import DataLoader
from enum import Enum

class ModelType(Enum):
    LSTM = 0
    CNN  = 1

# SETTINGS
MODEL_TYPE = ModelType.CNN
DATA_DIR = os.path.expanduser("./data/")
EMB_CACHE = os.path.expanduser("./")
DATASET_CACHE = os.path.expanduser("./")
MODEL_CHECKPOINTS = os.path.abspath('./')
BATCH_SIZE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
NUM_EPOCHS = 100
HIDDEN_SIZE = 8 # the LSTM's hidden/output size
NUM_WORKERS = 8
LEARNING_RATE = 0.001

if MODEL_TYPE == ModelType.LSTM:
    from model.lstm import Classifier
    from model.lstm import collate
elif MODEL_TYPE == ModelType.CNN:
    from model.cnn import Classifier
    from model.cnn import collate

def evaluate(dataloader, net):
    print("Evaluating... ", end="")
    correctCount = 0
    totalCount = 0
    for batchIdx, batch in enumerate(dataloader):
        # extract input and labels
        inputs, labels = batch['features'], batch['label']
        inputs = inputs.cuda() if DEVICE==torch.device('cuda') else inputs.cpu()
        labels = labels.to(DEVICE)

        # predict only
        predictions = net(inputs).to(DEVICE)

        # compute index of predicted class
        predClassIndices = torch.argmax(predictions, dim=1)

        # compute index of label class
        labelClassIndices = torch.argmax(labels, dim=1)

        # compute amount of correct predictions
        batchSize = len(labels)
        totalCount += batchSize
        for b in range(batchSize):
            correctCount += int(predClassIndices[b] == labelClassIndices[b])
    return float(correctCount)/float(totalCount)

def train(load=False, load_chkpt=None):
    # set up data
    wholeset, trainset, valset = dataset.loadAndSplitData(DATA_DIR+'train.json',
                                                          0.2)
    dataloaderTrain = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=0, collate_fn=collate)
    dataloaderVal   = DataLoader(valset, batch_size=len(valset.indices),
                                 shuffle=True, num_workers=0,
                                 collate_fn=collate)
    featureSize  = wholeset.numFeaturesPerFrame()
    labelSize    = wholeset.labelSize()
    #print(f"Feature size per frame: {featureSize}\nLabel size: {labelSize}")

    # set up model
    model = None
    if MODEL_TYPE == ModelType.LSTM:
        model = Classifier(inputSize=featureSize, outputSize=labelSize,
                           hiddenSize=HIDDEN_SIZE).to(DEVICE)
    elif MODEL_TYPE == ModelType.CNN:
        model = Classifier().to(DEVICE)

    # set up optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=0.9)

    metric_dict = {'loss': '------', 'accuracy': '------'}
    start_epoch = 0

    # load model from checkpoint
    if load:
        try:
            last_epoch, optimizer_state_dict, loss = model.load(MODEL_CHECKPOINTS, load_chkpt)
            start_epoch = last_epoch + 1
            optimizer.load_state_dict(optimizer_state_dict)
            metric_dict.update(loss=loss, accuracy=f'{100*evaluate(dataloaderVal, model):6.2f}%')
        except FileNotFoundError:
            pass

    # a nice progress bar to make the waiting time much better
    pbar = tqdm(total=NUM_EPOCHS*len(trainset), postfix=metric_dict)


    # run for NUM_EPOCHS epochs
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        # run for every data (in batches) of our iterator
        running_loss = 0.0

        pbar.set_description(f"Epoch {epoch + 1}/{start_epoch + NUM_EPOCHS}")
        for numBatch, batch in enumerate(dataloaderTrain):
            # extract input and labels
            inputs, labels = batch['features'], batch['label']
            inputs = inputs.cuda() if DEVICE==torch.device('cuda') else inputs.cpu()
            labels = labels.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            outputs = outputs.to(DEVICE)

            # 2D loss function expects input as (batch, prediction, sequence) and target as (batch, sequence) (containing the class INDEX)
            # loss = criterion(outputs.permute(0,2,1), labels)
            # otherwise use view function to get rid of sequence dimension by effectively concatenating all sequence items
            labels = torch.argmax(labels, dim=1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            pbar.update(labels.size(0))
            metric_dict.update({'loss': f'{loss.item():6.3f}'})
            pbar.set_postfix(metric_dict)

        #TODO later for dev: evaluate on training set after each epoch
        metric_dict.update({'accuracy': f'{100*evaluate(dataloaderVal, model):6.2f}%'})
        pbar.set_postfix(metric_dict)
        # save model
        model.save(MODEL_CHECKPOINTS, epoch, loss, optimizer)

def predict(filename, load=False, load_chkpt=None, num_samples=1, interactive=False):
    testData = dataset.LogMelDataset(filename)

def test():
    return True

if __name__ == "__main__":
    train(load=True)
    #predict(load=True, num_samples=10, interactive=True)
