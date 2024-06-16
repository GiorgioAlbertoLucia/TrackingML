'''
    Script to evaluate the performance of a Binary Classifier
'''
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch._dynamo
from torch.optim import lr_scheduler
import torch
torch._dynamo.config.suppress_errors = True

import sys
sys.path.append('../..')
from utils.terminal_colors import TerminalColors as tc
from utils.timeit import timeit

from core.fcnn import FullyConnectedClassifier, ClassifierHandler
from core.losses import WeightedBCELoss
from core.data_handler import BinaryHandler


@timeit
def train_and_test(train_bh: BinaryHandler, test_bh: BinaryHandler, threshold: float = 0.85, **kwargs):
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
    scheduler_hard_mining = lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.1)

    train_bh.become_eager()
    test_bh.become_eager()
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
        ch.model.load_state_dict(torch.load(model_path))

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

            scheduler_hard_mining.step()

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

    if True:
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
