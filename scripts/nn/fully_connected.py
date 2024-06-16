'''
    Script to evaluate the performance of a Binary Classifier
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch._dynamo
import torch.nn as nn
from torch.optim import lr_scheduler
import torch
torch._dynamo.config.suppress_errors = True

import sys
sys.path.append('../..')
from utils.terminal_colors import TerminalColors as tc
from utils.timeit import timeit

from core.track_reconstruction import TrackReconstruction
from core.fcnn import FullyConnectedClassifier, ClassifierHandler
from core.losses import WeightedBCELoss
from core.data_handler import BinaryHandler

from data_loading import data_loading
from logistic_regression import logistic_regression
from train_and_test import train_and_test
from track_reconstruction import track_reconstruction

if __name__ == '__main__':

    # Load data
    train_bh, test_bh = data_loading(
        n_particles=100, train_pair='load', test_pair='load')

    # Train and evaluate Logistic Regression HARD_NEGATIVE_model
    logistic_regression(train_bh, test_bh, evaluate_test_accuracy=True)

    # Train and evaluate FullyConnectedClassifier neural network
    ch = train_and_test(train_bh, test_bh, do_training=False, do_hard_negative_mining=False,
                         build_negative_mining=False, evaluate_test_accuracy=True, threshold=0.85)

    # Perform track reconstruction
    track_reconstruction(train_bh, test_bh, ch, reconstruct_training=False, load_reco=False)
