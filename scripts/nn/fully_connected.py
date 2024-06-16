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

from data_loading import data_loading as new_data_loading

@timeit
def data_loading(n_particles: int = None, test_pair: str = 'load'):
    '''
        Load data for the train and test sets.
        Reduce the test set to n_particles (if n is not None)
    '''

    print(tc.GREEN+tc.BOLD+'Loading data...'+tc.RESET)
    data_dir = '../../data/train_1'
    detector_file = '../../data/detectors.csv'

    # Training set
    train_events = [f'0000010{ev_idx}' for ev_idx in range(10, 20)]
    train_bh = BinaryHandler(data_dir, train_events,
                             opt='train', detector_file=detector_file)
    #train_bh.build_negative_dataset()
    print(train_bh)

    # Test set
    test_events = ['000001001']
    test_bh = BinaryHandler(data_dir, test_events,
                            opt='test', detector_file=detector_file)
    if n_particles is not None:
        test_bh.reduce_to_n_particles(n_particles)
    print(test_bh)

    if test_pair == 'load':
        test_bh.load_pair_dataset('../../data/save/test_pairs.npy')
    elif test_pair == 'build':
        test_bh.build_pair_dataset()
        test_bh.save_pair_dataset('../../data/save/test_pairs.npy')

    # Feature scaling
    train_mean = train_bh.Xmean
    train_std = train_bh.Xstd

    train_bh.feature_scaling(train_mean, train_std)
    test_bh.feature_scaling(train_mean, train_std)
    test_bh.save_pair_dataset('../../data/save/test_pairs_scaled.npy')

    del train_mean, train_std
    return train_bh, test_bh

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

@timeit
def fully_connected(train_bh: BinaryHandler, test_bh: BinaryHandler, threshold: float = 0.85, **kwargs):
    '''
        Initialize a FullyConnectedClassifier neural network
    '''

    print(tc.GREEN+tc.BOLD +
          '\nInitializing FullyConnectedClassifier neural network...'+tc.RESET)

    INPUT_SIZE = 12
    LEARNING_RATE = 1e-2
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    FORCE_CPU = True
    WEIGHT_FALSE_POS = 1.
    WEIGHT_FALSE_NEG = 50.

    fc_model = FullyConnectedClassifier(INPUT_SIZE)
    loss_function = WeightedBCELoss(weight_fp=WEIGHT_FALSE_POS, weight_fn=WEIGHT_FALSE_NEG)
    #optimizer = torch.optim.SGD(
    optimizer = torch.optim.Adam(
        fc_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) #, momentum=MOMENTUM)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    #scheduler_hard_mining = lr_scheduler.StepLR(
    #    optimizer, step_size=4, gamma=0.1)

    #train_bh.become_eager()
    #test_bh.become_eager()
    ch = ClassifierHandler(fc_model, loss_function,
                           optimizer, force_cpu=FORCE_CPU)
    ch.load_data(train_bh, test_bh)

    # Training
    EPOCHS = 48
    ACCUMULATION_STEPS = 2
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    model_path = '../../models/fc_model_state_dict.pth'
    model_path_hard_mining = '../../models/fc_model_state_dict_hard_mining.pth'

    HARD_NEGATIVE_EPOCHS = 20

    if kwargs.get('do_training', False):
        print(tc.GREEN+tc.BOLD +
              '\nTraining FullyConnectedClassifier neural network...'+tc.RESET)
        for epoch in tqdm(range(EPOCHS)):

            # Train
            train_loss = ch.train(ACCUMULATION_STEPS)
            train_losses.append(train_loss)
            print(f'Epoch {epoch+1}/{EPOCHS}')
            print(f'\t- train loss: {train_loss:.4f}')

            # Evaluate
            if (epoch) % 8 == 0:
                train_accuracy, train_precision, train_sensitivity = ch.evaluate(
                    opt='train', threshold=threshold)
                train_accuracies.append(train_accuracy)
                print(f'\t- train accuracy: {train_accuracy:.8f}')
                print('\t- Train precision: ', train_precision)
                print('\t- Train sensitivity: ', train_sensitivity)

                if kwargs.get('evaluate_test_accuracy', False):
                    test_accuracy, test_precision, test_sensitivity = ch.evaluate(
                        opt='test', threshold=threshold)
                    test_accuracies.append(test_accuracy)
                    print(f'\t- test accuracy: {test_accuracy:.8f}')
                    print('\t- Train precision: ', test_precision)
                    print('\t- Train sensitivity: ', test_sensitivity)

            scheduler.step()

        torch.save(ch.model.state_dict(), model_path)

        # Plot training results
        epochs = [iepoch for iepoch in range(EPOCHS)]

        plt.figure(figsize=(12, 4))
        plt.plot(epochs, train_losses, label="Train")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.savefig('../../plots/fc_loss.png')

        epochs = [iepoch*4 for iepoch in range(EPOCHS//8)]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(epochs, train_accuracies, label="Train")
        axs[0].set_title(f'Train')
        if kwargs.get('evaluate_test_accuracy', False):
            axs[1].plot(epochs, test_accuracies, label="Test")
            axs[1].set_title(f'Test')
        for ax in axs:
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.legend()
        plt.savefig('../../plots/fc_accuracy.png')

    else:
        print(tc.GREEN+tc.BOLD +
              '\nLoading FullyConnectedClassifier neural network...'+tc.RESET)
        #ch.model.load_state_dict(torch.load(model_path))

    # Hard negative mining
    if kwargs.get('do_hard_negative_mining', False):
        print(tc.GREEN+tc.BOLD+'\nHard negative mining...'+tc.RESET)

        if kwargs.get('build_negative_mining', False):
            train_bh.hard_negative_mining(
                ch.model, device=ch.device, threshold=0.5, mining_size=int(3*1e6), batch_size=int(5*1e5))
            train_bh.save_pair_dataset(
                '../../data/save/train_pairs_hard_negative.npy')
        else:
            train_bh.load_pair_dataset(
                '../../data/save/train_pairs_hard_negative.npy')

        # reload data
        ch.load_data(train_bh, test_bh)

        for epoch in tqdm(range(HARD_NEGATIVE_EPOCHS)):

            # Train
            train_loss = ch.train(ACCUMULATION_STEPS)
            train_losses.append(train_loss)
            print(f'Epoch {epoch+1}/{HARD_NEGATIVE_EPOCHS}')
            print(f'\t- train loss: {train_loss:.4f}')

            # Evaluate
            if (epoch) % 4 == 0:
                train_accuracy, __, __ = ch.evaluate(
                    opt='train', threshold=threshold)
                train_accuracies.append(train_accuracy)
                print(f'\t- train accuracy: {train_accuracy:.8f}')

                if kwargs.get('evaluate_test_accuracy', False):
                    test_accuracy, __, __ = ch.evaluate(
                        opt='test', threshold=threshold)
                    test_accuracies.append(test_accuracy)
                    print(f'\t- test accuracy: {test_accuracy:.8f}')

            #scheduler_hard_mining.step()

        torch.save(ch.model.state_dict(), model_path_hard_mining)

        # Plot training results
        epochs = [iepoch for iepoch in range(HARD_NEGATIVE_EPOCHS)]

        plt.figure(figsize=(12, 4))
        plt.plot(epochs, train_losses, label="Train")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.savefig('../../plots/fc_hard_negative_loss.png')

        epochs = [iepoch*4 for iepoch in range(HARD_NEGATIVE_EPOCHS//4)]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(epochs, train_accuracies, label="Train")
        axs[0].set_title(f'Train')
        if kwargs.get('evaluate_test_accuracy', False):
            axs[1].plot(epochs, test_accuracies, label="Test")
            axs[1].set_title(f'Test')
        for ax in axs:
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Accuracy")
            ax.legend()
        plt.savefig('../../plots/fc_hard_negative_accuracy.png')

    # Load one of the models
    print(tc.GREEN+tc.BOLD +
          '\nLoading FullyConnectedClassifier neural network...'+tc.RESET)
    ch.model.load_state_dict(torch.load(model_path))
    #ch.model.load_state_dict(torch.load(model_path_hard_mining))

    if False:
        # Evaluate accuracy
        print(tc.GREEN+tc.BOLD +
              '\nEvaluating FullyConnectedClassifier neural network...'+tc.RESET)

        train_acc, train_precision, train_sensitivity, y_outs, y_preds = ch.evaluate(
            opt='train', threshold=threshold, save_predictions=True)
        print('\t- Train accuracy: ', train_acc)
        print('\t- Train precision: ', train_precision)
        print('\t- Train sensitivity: ', train_sensitivity)
        train_bh.pair_dataset = np.hstack(
            (train_bh.pair_dataset, y_outs, y_preds))
        train_bh.save_pair_dataset('../../data/save/train_pred_set.npy')

        del train_bh

        if kwargs.get('evaluate_test_accuracy', False):
            test_acc, test_preccision, test_sensitivity, y_outs, y_preds = ch.evaluate(
                opt='test', threshold=threshold, save_predictions=True)
            print('\t- Test accuracy: ', test_acc)
            print('\t- Test precision: ', test_preccision)
            print('\t- Test sensitivity: ', test_sensitivity)
            test_bh.pair_dataset = np.hstack(
                (test_bh.pair_dataset, y_outs, y_preds))
            test_bh.save_pair_dataset('../../data/save/test_pred_set.npy')

    return ch

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


if __name__ == '__main__':

    # Load data
    #train_bh, test_bh = new_data_loading(
    #    n_particles=100, train_pair='load', test_pair='load')
    train_bh, test_bh = data_loading(
        n_particles=30, test_pair='load')

    # Train and evaluate Logistic Regression HARD_NEGATIVE_model
    #logistic_regression(train_bh, test_bh, evaluate_test_accuracy=True)

    # Train and evaluate FullyConnectedClassifier neural network
    ch = fully_connected(train_bh, test_bh, do_training=False, do_hard_negative_mining=False,
                         build_negative_mining=False, evaluate_test_accuracy=True, threshold=0.85)

    # Perform track reconstruction
    track_reconstruction(train_bh, test_bh, ch, reconstruct_training=False, load_reco=False)
