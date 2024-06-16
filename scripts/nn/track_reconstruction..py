'''
    Script to evaluate the performance of a Binary Classifier
'''

import torch._dynamo
import torch
torch._dynamo.config.suppress_errors = True

import sys
sys.path.append('../..')
from utils.terminal_colors import TerminalColors as tc
from utils.timeit import timeit

from core.track_reconstruction import TrackReconstruction
from core.fcnn import ClassifierHandler
from core.data_handler import BinaryHandler

from data_loading import data_loading

@timeit
def track_reconstruction(train_bh: BinaryHandler, test_bh: BinaryHandler, ch: ClassifierHandler, reconstruct_training: bool = False, load_reco: bool = False):
    '''
        Perform track reconstruction using the FullyConnectedClassifier neural network
    '''

    print(tc.GREEN+tc.BOLD +
          '\nInitializing track reconstruction neural network...'+tc.RESET)
    fc_model = ch.model
    device = ch.device

    # reconstruct tracks (train set)
    if reconstruct_training:
        tr_train = TrackReconstruction(train_bh, fc_model, device)
        tr_train.compute_prediction_matrix(
            minimum_threshold=0.5, output_file='../../data/save/train_prediction_matrix.npy')
        tr_train.reconstruct_all_tracks(
            threshold=0.85, output_file='../../data/save/train_all_tracks.npy')
        tr_train.score_tracks()
        tr_train.select_tracks()
        tr_train.evaluate_tracking()

    # reconstruct tracks (test set)

    tr_test = TrackReconstruction(test_bh, fc_model, device)
    if load_reco:
        tr_test.load_prediction_matrix(
            '../../data/save/test_prediction_matrix.npy')
        tr_test.load_all_tracks('../../data/save/test_all_tracks.npy')
    else:
        tr_test.compute_prediction_matrix(
            minimum_threshold=0.5, output_file='../../data/save/test_prediction_matrix.npy')
        tr_test.reconstruct_all_tracks(
            threshold=0.8, output_file='../../data/save/test_all_tracks.npy')
    tr_test.score_tracks()
    tr_test.select_tracks()

    #input_dataframe_file = '../../data/save/test_reco_tracks.csv'
    input_dataframe_file = None
    tr_test.evaluate_tracking(
        event=1001, input_dataframe_file=input_dataframe_file)
    tr_test.save_tracks(output_file='../../data/save/test_reco_tracks.csv')
