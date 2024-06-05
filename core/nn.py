'''
    Neural Network classes
'''
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from .data_handler import BinaryHandler

class FullyConnectedClassifier(nn.Module):

    def __init__(self, input_dimension:int, sigmoid_output:bool=True):
        '''
            Class to create a fully connected neural network architecture 
            that processes the input into a binary classification output
        '''
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 32)
        #self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

        self.non_linearity = F.relu
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        x = self.non_linearity(self.fc1(x))
        #x = self.non_linearity(self.fc2(x))
        x = self.non_linearity(self.fc3(x))
        x = self.non_linearity(self.fc4(x))
        if self.sigmoid_output: return torch.sigmoid(x)
        return x
    

# Class to initialise and train the neural network
class ClassifierHandler:

    def __init__(self, model:FullyConnectedClassifier, loss_function, optimizer, **kwargs):

        self._load_device(kwargs.get('force_cpu', False))

        self.model = model
        self.loss_function = loss_function  
        self.optimizer = optimizer

        self.train_loader = None
        self.test_loader = None

    def _load_device(self, force_cpu:bool=False) -> None:
        '''
            Load the device to run the model on
        '''

        # Check for GPU
        if torch.backends.mps.is_available():
           self.device = torch.device("mps")
        else:
           print ("MPS device not found.")
           force_cpu = True

        if force_cpu:
            self.device = torch.device("cpu")

        x = torch.ones(1, device=self.device)
        print(x)
        del x

    def load_data(self, train_bh:BinaryHandler, test_bh:BinaryHandler):
        '''
            Load the data into the model
        '''

        print('Loading data into the model...')
        print(f'Unique labels in test: {np.unique(test_bh.y)}')

        # Create a WeightedRandomSampler for the test data
        class_sample_count = np.unique(test_bh.y, return_counts=True)[1]
        weights = 1. / torch.tensor(class_sample_count, dtype=torch.float)
        sample_weights = weights[test_bh.y]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        self.train_loader = torch.utils.data.DataLoader(train_bh, batch_size=64, shuffle=True, drop_last=False,
                                           pin_memory=True, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(test_bh, batch_size=64, shuffle=False, drop_last=False,
                                            pin_memory=True, num_workers=4)
        
        # Collect all labels from test_loader
        all_labels = []
        for _, labels in self.test_loader:
            all_labels.extend(labels.tolist())
            del labels

        # Print unique labels in test_loader
        print(f'Unique labels in test_loader: {np.unique(all_labels)}')
        del all_labels
    
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
        return total_loss / len(self.train_loader)
    
    def evaluate(self, opt:str, threshold:float):
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
        
        self.model.sigmoid_output = True
        self.model.eval()

        y_true = []
        y_pred = []

        for ibatch, (X_batch, y_batch) in tqdm(enumerate(data_loader)):

            if ibatch < 5:
                print(f"Batch {ibatch} labels: {y_batch}")

            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            y_pred_batch = self.model(X_batch).squeeze()
            y_pred_batch = (y_pred_batch > threshold).float()
            y_true.extend(y_batch.tolist())
            y_pred.extend(y_pred_batch.tolist())
            del X_batch, y_batch, y_pred_batch

        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        accuracy = (y_true == y_pred).float().mean()
        precision = ((y_true == 1.) & (y_pred == 1.)).sum().float() / y_pred.sum()  
        sensitivity = ((y_true == 1.) & (y_pred == 1.)).sum().float() / y_true.sum()

        check = np.hstack((y_true.reshape(len(y_true), 1), y_pred.reshape(len(y_true), 1)))
        np.save(f'../../data/{opt}_check.npy', check)
        
        self.model.sigmoid_output = False
        del check
        del y_true, y_pred
        return accuracy, precision, sensitivity
    

class Convolutional(nn.Module):

    def __init__(self, input_dimension:int, sigmoid_output:bool=True):
        
        raise NotImplementedError('Convolutional not implemented yet')