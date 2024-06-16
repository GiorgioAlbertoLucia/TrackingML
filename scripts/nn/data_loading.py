import torch._dynamo
import torch
torch._dynamo.config.suppress_errors = True

import sys
sys.path.append('../..')
from utils.terminal_colors import TerminalColors as tc
from utils.timeit import timeit

from core.data_handler import BinaryHandler


@timeit
def data_loading(n_particles: int = None, train_pair: str = 'load', test_pair: str = 'load'):
    '''
        Load data for the train and test sets.
        Reduce the test set to n_particles (if n is not None)
    '''

    print(tc.GREEN+tc.BOLD+'Loading data...'+tc.RESET)
    data_dir = '../../data/train_1'
    detector_file = '../../data/detectors.csv'
    save_dir = '../../data/save'

    # Training set
    train_events = [f'0000010{ev_idx}' for ev_idx in range(10, 15)]
    train_bh = BinaryHandler(data_dir, train_events, save_dir,
                             opt='train', mode=train_pair, detector_file=detector_file)
    print(train_bh)

    # Test set
    test_events = ['000001001']
    test_bh = BinaryHandler(data_dir, test_events, save_dir,
                            opt='test', mode=test_pair, n_particles=n_particles, detector_file=detector_file)
    print(test_bh)

    # Feature scaling
    train_mean = 0.
    train_std = 1.

    if train_pair == 'build' or test_pair == 'build':
        train_mean = train_bh.Xmean
        train_std = train_bh.Xstd

    if train_pair == 'build':
        train_bh.feature_scaling(train_mean, train_std)
    if test_pair == 'build':
        test_bh.feature_scaling(train_mean, train_std)

    print(tc.GREEN+tc.BOLD+'Data loaded!'+tc.RESET)

    del train_mean, train_std
    print('cleaning up...')
    return train_bh, test_bh
