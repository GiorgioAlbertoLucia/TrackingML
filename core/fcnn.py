'''
    Fully Connected Neural Network classes
'''
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from .data_handler import BinaryHandler


class FullyConnectedClassifier(nn.Module):

    def __init__(self, input_dimension: int, sigmoid_output: bool = True):
        '''
            Class to create a fully connected neural network architecture 
            that processes the input into a binary classification output
        '''
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 54)
        self.fc4 = nn.Linear(54, 1)

        self.activation = F.relu
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)
        


# Class to initialise and train the neural network
class ClassifierHandler:

    def __init__(self, model: FullyConnectedClassifier, loss_function, optimizer, **kwargs):

        self._load_device(kwargs.get('force_cpu', False))

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.train_loader = None
        self.test_loader = None

    def _load_device(self, force_cpu: bool = False) -> None:
        '''
            Load the device to run the model on
        '''

        # Check for GPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            print("MPS device not found.")
            force_cpu = True

        if force_cpu:
            self.device = torch.device("cpu")

        x = torch.ones(1, device=self.device)
        print(x)
        del x

    def load_data(self, train_bh: BinaryHandler, test_bh: BinaryHandler):
        '''
            Load the data into the model
        '''

        print('Loading data into the model...')

        BATCH_SIZE = 5000

        self.train_loader = DataLoader(train_bh, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
                                       pin_memory=True, num_workers=4)
        test_ds = TensorDataset(torch.tensor(
            test_bh.X, dtype=torch.float32), torch.tensor(test_bh.y, dtype=torch.float32))
        self.test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
                                      pin_memory=True, num_workers=4)

    def train(self, accumulation_steps=1):
        '''
            Train the model for a single epoch
        '''

        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        for ibatch, (X_batch, y_batch) in enumerate(self.train_loader):
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            y_pred = self.model(X_batch).squeeze()
            loss = self.loss_function(y_pred, y_batch)
            loss.backward()
            if (ibatch + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item()
            del X_batch, y_batch, y_pred, loss
            torch.cuda.empty_cache()
        return total_loss / len(self.train_loader)

    def evaluate(self, opt: str, threshold: float, save_predictions: bool = False):
        '''
            Evaluate the accuracy, precision and sensitivity of the model on the data_loader.
            * Accuracy is defined as the ratio of correct predictions to the total number of predictions.
            * Precision is defined as the ratio of true positives to the sum of true positives and false positives.
            * Sensitivity is defined as the ratio of true positives to the sum of true positives and false negatives.
            A sigmoid is applied to the output of the model to get the prediction.
        '''

        data_loader = None
        if opt == 'train':
            data_loader = self.train_loader
        elif opt == 'test':
            data_loader = self.test_loader
        else:
            raise ValueError('Invalid option. Choose either "train" or "test"')

        self.model.eval()

        y_true = []
        y_out = []
        y_pred = []

        with torch.no_grad():
            for (X_batch, y_batch) in tqdm(data_loader):

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_out_batch = self.model(X_batch).squeeze()
                y_pred_batch = (y_out_batch > threshold).float()
                y_true.extend(y_batch.tolist())
                y_out.extend(y_out_batch.float().tolist())
                y_pred.extend(y_pred_batch.tolist())
                del X_batch, y_batch, y_out_batch, y_pred_batch

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        y_out = torch.tensor(y_out)

        accuracy = (y_true == y_pred).float().mean()
        precision = ((y_true == 1.) & (y_pred == 1.)
                     ).sum().float() / y_pred.sum()
        sensitivity = ((y_true == 1.) & (y_pred == 1.)
                       ).sum().float() / y_true.sum()
        
        if save_predictions:
            predictions_ds = np.hstack((y_true.numpy().reshape(-1, 1), y_pred.numpy().reshape(-1, 1), y_out.numpy().reshape(-1, 1)))
            np.save(f'../../data/save/{opt}_pred_set.npy', predictions_ds)

        del y_true, y_pred
        return accuracy, precision, sensitivity


class Convolutional(nn.Module):

    def __init__(self, input_dimension: int, sigmoid_output: bool = True):

        raise NotImplementedError('Convolutional not implemented yet')
