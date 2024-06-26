'''
    Fully Connected Neural Network classes
'''
import os
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from .data_handler import BinaryHandler


class FullyConnectedClassifier(nn.Module):

    def __init__(self, input_dimension: int):
        '''
            Class to create a fully connected neural network architecture 
            that processes the input into a binary classification output
        '''
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.activation = F.relu

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

    def _append_to_memmap(self, data, filename, shape_file):

        new_shape = None
        if not os.path.exists(filename):
            fp = np.memmap(filename, dtype='float32', mode='w+', shape=data.shape)
            fp[:] = data[:]
            new_shape = data.shape

        else:
            with open(shape_file, 'r') as f:
                current_size = json.load(f)[0]
            old_data = np.fromfile(filename, dtype='float32').reshape((current_size, data.shape[1]))
            new_shape = (current_size + data.shape[0], data.shape[1])
            fp = np.memmap(filename, dtype='float32', mode='w+', shape=new_shape)
            fp[:current_size] = old_data[:]
            fp[current_size:] = data[:]

        del fp

        with open(shape_file, 'w') as f:
            json.dump(new_shape, f)

    def load_data(self, train_bh: BinaryHandler, test_bh: BinaryHandler):
        '''
            Load the data into the model
        '''

        print('Loading data into the model...')

        #BATCH_SIZE_TRAIN = 1000     # use different batch sizes for training and testing
        #BATCH_SIZE_TRAIN = 256     # use different batch sizes for training and testing
        BATCH_SIZE_TRAIN = 100000   # use different batch sizes for training and testing
        BATCH_SIZE_TEST = 100000

        self.train_loader = DataLoader(train_bh, batch_size=BATCH_SIZE_TRAIN, shuffle=False, drop_last=False,
                                       pin_memory=False, num_workers=4)
        self.test_loader = DataLoader(test_bh, batch_size=BATCH_SIZE_TEST, shuffle=False, drop_last=False,
                                      pin_memory=False, num_workers=3)

    def train(self, accumulation_steps=1):
        '''
            Train the model for a single epoch
        '''

        self.model.train()
        total_loss = 0
        self.optimizer.zero_grad()
        for ibatch, (X_batch, y_batch) in tqdm(enumerate(self.train_loader)):
            #print(f'batch {ibatch}')
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            y_pred = self.model(X_batch).squeeze()
            loss = self.loss_function(y_pred, y_batch)
            loss.backward()
            if (ibatch + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item()
            del X_batch, y_batch, y_pred, loss
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

        accuracy_list = []
        precision_list = []
        sensitivity_list = []

        null_y_pred = 0
        null_y_true = 0

        y_outs = []
        y_preds = []

        filename = f'../../data/save/{opt}_pred_set.npy'
        shape_file = f'../../data/save/{opt}_pred_set_shape.json'
        os.remove(filename) if os.path.exists(filename) else None

        with torch.no_grad():
            for (X_batch, y_batch) in tqdm(data_loader):

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_out_batch = self.model(X_batch).squeeze()
                y_pred_batch = (y_out_batch > threshold).float()

                if save_predictions:

                    y_outs.append(y_out_batch.cpu().numpy().reshape(-1, 1))
                    y_preds.append(y_pred_batch.cpu().numpy().reshape(-1, 1))

                y_true = y_batch
                y_pred = y_pred_batch

                accuracy = (y_true == y_pred).float().mean()
                precision = 0.
                if y_pred.sum() == 0:
                    null_y_pred += 1
                else:
                    precision = ((y_true == 1.) & (y_pred == 1.)).sum().float() / y_pred.sum()
                sensitivity = 0.
                if y_true.sum() == 0:
                    null_y_true += 1
                else:
                    sensitivity = ((y_true == 1.) & (y_pred == 1.)).sum().float() / y_true.sum()

                accuracy_list.append(accuracy)
                precision_list.append(precision)
                sensitivity_list.append(sensitivity)


                del X_batch, y_batch, y_out_batch, y_pred_batch
        
        if save_predictions:
            y_outs = np.vstack(y_outs)
            y_preds = np.vstack(y_preds)
        
        print(f'Null y_pred fraction: {null_y_pred / len(data_loader)}')
        print(f'Null y_true fraction: {null_y_true / len(data_loader)}')
        
        if save_predictions:
            return np.mean(accuracy_list), np.mean(precision_list), np.mean(sensitivity_list), y_outs, y_preds    
        return np.mean(accuracy_list), np.mean(precision_list), np.mean(sensitivity_list)


