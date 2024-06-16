'''
    Script to train a Logistic Regression model
'''

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import torch._dynamo
import torch
torch._dynamo.config.suppress_errors = True

import sys
sys.path.append('../..')
from utils.terminal_colors import TerminalColors as tc
from utils.timeit import timeit

from core.data_handler import BinaryHandler

@timeit
def logistic_regression(train_bh: BinaryHandler, test_bh: BinaryHandler, evaluate_test_accuracy: bool = False):
    '''
        Train a Logistic Regression model
    '''

    print(tc.GREEN+tc.BOLD+'\nTraining Logistic Regression model...'+tc.RESET)
    lr = LogisticRegression()
    lr.fit(train_bh.X, train_bh.y)

    print(tc.GREEN+tc.BOLD+'Evaluating Logistic Regression model...'+tc.RESET)
    # Training set accuracy
    train_y_pred = lr.predict(train_bh.X)
    print(
        f'\t- Train accuracy: {accuracy_score(train_bh.y, train_y_pred):.4f}')

    # Test set accuracy
    if evaluate_test_accuracy:
        test_y_pred = lr.predict(test_bh.X)
        print(
            f'\t- Test accuracy: {accuracy_score(test_bh.y, test_y_pred):.4f}')
