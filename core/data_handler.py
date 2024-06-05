'''
    Class for data processing
'''

import numpy as np
import numba
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset
import torch

import sys
sys.path.append('..')
from utils.terminal_colors import TerminalColors as tc



class DataHandler(Dataset):

    def __init__(self, event_numbers: str, cell_csvs: str, hits_csvs: str, truth_csvs: str):

        cell_dfs = []
        for ev_idx, event_number in enumerate(event_numbers):
            cell_df = pd.read_csv(cell_csvs[ev_idx])
            cell_df['event_id'] = int(event_number)
            cell_dfs.append(cell_df)
        hits_dfs = [pd.read_csv(hits_csv) for hits_csv in hits_csvs]
        truth_dfs = [pd.read_csv(truth_csv) for truth_csv in truth_csvs]

        cell_df = pd.concat(cell_dfs, ignore_index=True)
        hits_df = pd.concat(hits_dfs, ignore_index=True)
        truth_df = pd.concat(truth_dfs, ignore_index=True)

        tmp_df = pd.merge(hits_df[['hit_id', 'x', 'y', 'z', 'volume_id']], truth_df[[
                          'hit_id', 'particle_id']], on='hit_id')
        self.dataset = pd.merge(
            tmp_df, cell_df[['hit_id', 'value', 'event_id']], on='hit_id')

        self.x: torch.tensor = torch.tensor(0., dtype=float)
        self.y: torch.tensor = torch.tensor(0., dtype=float)

        self._update()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert 0 <= idx < len(
            self.x), f"Index {idx} out of bounds for dataset of size {len(self.x)}"
        return self.x[idx], self.y[idx]

    # create a print function to print information about the DataHandler object
    def __str__(self):
        return f'DataHandler object.\nDataset:\n\t- type: {type(self.dataset)}\n\t- length: {len(self.dataset)}\n\t- columns: {self.columns}\nx: \n\t- shape: {self.x.shape}\n\t- type: {type(self.x)}\ny: \n\t- shape: {self.y.shape}\n\t- type: {type(self.y)}\n'

    @property
    def columns(self):
        return ['x', 'y', 'z', 'value', 'event_id'], ['particle_id']

    @property
    def features(self):
        return ['x', 'y', 'z', 'value']

    def mean(self):
        mean = self.dataset[self.columns[0]].mean()
        mean['event_id'] = 0
        return mean

    def std(self):
        std = self.dataset[self.columns[0]].std()
        std['event_id'] = 1
        return std

    def feature_scaling(self, mean, std):
        input_features, __ = self.columns
        self.dataset[input_features] = (
            self.dataset[input_features] - mean) / std
        self._update()

    def _update(self):
        '''
            Update the feature and label tensor after the dataset is modified.
        '''

        self.x = torch.tensor(
            self.dataset[self.features].values, dtype=torch.float32)
        self.y = torch.tensor(
            self.dataset['particle_id'].values, dtype=torch.float32)

    def reduce_to_n_particles(self, n_particles: int):
        '''
            Reduce the dataset to only contain the first n_particles
        '''

        selected_particles = self.dataset['particle_id'].unique()[:n_particles]
        self.dataset = self.dataset[self.dataset['particle_id'].isin(
            selected_particles)]
        self._update()

    def reduce_to_inner_barrel(self):
        '''
            Reduce the dataset to only contain the inner barrel hits
        '''

        self.dataset = self.dataset.query(
            'volume_id==7 or volume_id==8 or volume_id==9')
        self._update()


class BinaryHandler(Dataset):

    def __init__(self, data_dir: str, events: list, opt: str = 'train'):
        '''
            Class to create a dataset from the csv files for the binary classification task

            Parameters:
            - data_dir: directory where the csv files are stored
            - events: list of event numbers to be processed
            - type: type of the dataset (train or test)

            pair_dataset: numpy array containing the dataset for the binary classification task
            dataset: numpy array used for tracking
                * x: hit coordinate
                * y: hit coordinate
                * z: hit coordinate
                * cluster_size: number of hits in the cluster
                * amplitude: total energy deposited in the cluster
                * particle_id: particle id
                * event_id: event id
        '''

        self.pair_dataset = None
        self.dataset = None
        self.events = []

        self.opt = opt

        self.PREDICTION_FEATURES = 5
        self.PID_IDX = 5
        self.WEIGHTS_IDX = 6
        self.EVENT_IDX = 7

        if opt == 'train':
            self._build_train(data_dir, events)
        elif opt == 'test':
            self._build_test(data_dir, events)
        else:
            raise ValueError(f'Invalid dataset type: {opt}')

    def _build_train(self, data_dir: str, events: list):
        '''
            Build the training dataset
        '''

        print('Building training dataset...')

        for ev in events:
            print(f'Processing event {ev}')

            hits = pd.read_csv(f'{data_dir}/event{ev}-hits.csv')
            cells = pd.read_csv(f'{data_dir}/event{ev}-cells.csv')
            truth = pd.read_csv(f'{data_dir}/event{ev}-truth.csv')
            particles = pd.read_csv(f'{data_dir}/event{ev}-particles.csv')

            # store cluster size in a numpy array
            cl_size = cells.groupby(['hit_id'])['value'].count().values
            # store total energy deposited in a numpy array
            amplitude = cells.groupby(['hit_id'])['value'].sum().values

            num_ev = float(int(ev.lstrip('0')))
            event_id = np.ones((len(hits), 1))*int(num_ev)
            self.events.append(num_ev)

            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
                len(amplitude), 1), truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1)), dtype=np.float32)
            particle_ids = truth['particle_id'].unique()
            # remove noise hits from the particle ids
            particle_ids = particle_ids[np.where(particle_ids != 0)[0]]

            pair_list = []
            for p_id in particle_ids:
                hit_ids = truth[truth['particle_id'] == p_id]['hit_id'].values - 1
                id1, id2 = np.meshgrid(hit_ids, hit_ids)
                pairs = np.column_stack((id1.ravel(), id2.ravel()))
                pair_list.append(pairs[pairs[:, 0] != pairs[:, 1]])

            pairs = np.vstack(pair_list)

            features_idx1 = feature_dataset[pairs[:,
                                                  0], :self.PREDICTION_FEATURES]
            features_idx2 = feature_dataset[pairs[:,
                                                  1], :self.PREDICTION_FEATURES]
            pair_dataset = np.hstack((features_idx1, features_idx2, np.ones(
                (features_idx1.shape[0], 1)))).astype(np.float32)

            if self.pair_dataset is None:
                self.pair_dataset = pair_dataset
            else:
                self.pair_dataset = np.vstack(
                    (self.pair_dataset, pair_dataset))

            # negative samples

            NEGATIVE_SAMPLE_RATE = 3
            n = len(hits)
            size = len(feature_dataset)*NEGATIVE_SAMPLE_RATE
            p_id = truth['particle_id'].values
            id1_list = np.random.randint(n, size=size)
            id2_list = np.random.randint(n, size=size)
            pair = np.hstack((id1_list.reshape(size, 1),
                             id2_list.reshape(size, 1)))
            pair = pair[((p_id[id1_list] == 0) | (
                p_id[id1_list] != p_id[id2_list]))]

            features_idx1 = feature_dataset[pair[:,
                                                 0], :self.PREDICTION_FEATURES]
            features_idx2 = feature_dataset[pair[:,
                                                 1], :self.PREDICTION_FEATURES]
            negative_dataset = np.hstack((features_idx1, features_idx2, np.zeros(
                (features_idx1.shape[0], 1)))).astype(np.float32)
            self.pair_dataset = np.vstack(
                (self.pair_dataset, negative_dataset), dtype=np.float32)

            if self.dataset is None:
                self.dataset = feature_dataset
            else:
                self.dataset = np.vstack((self.dataset, feature_dataset))

            del hits, cells, truth, particles
            del feature_dataset, negative_dataset

        np.random.shuffle(self.pair_dataset)

    def _build_test(self, data_dir: str, events: list):
        '''
            Build the test dataset
        '''

        print('Building test dataset...')

        for ev in events:
            print(f'Processing event {ev}')

            hits = pd.read_csv(f'{data_dir}/event{ev}-hits.csv')
            cells = pd.read_csv(f'{data_dir}/event{ev}-cells.csv')
            truth = pd.read_csv(f'{data_dir}/event{ev}-truth.csv')
            particles = pd.read_csv(f'{data_dir}/event{ev}-particles.csv')

            # store cluster size in a numpy array
            cl_size = cells.groupby(['hit_id'])['value'].count().values
            # store total energy deposited in a numpy array
            amplitude = cells.groupby(['hit_id'])['value'].sum().values

            num_ev = float(int(ev.lstrip('0')))
            event_id = np.ones((len(hits), 1))*int(num_ev)
            self.events.append(num_ev)

            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
                len(amplitude), 1), truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1)), dtype=np.float32)

            if self.dataset is None:
                self.dataset = feature_dataset
            else:
                self.dataset = np.vstack((self.dataset, feature_dataset))

            del hits, cells, truth, particles
            del feature_dataset

    def build_pair_dataset(self, batch_size: int = 10000):
        '''
           Build the pair dataset for the test dataset.
        '''

        print('Building pair dataset...')

        if self.opt != 'test':
            raise ValueError(
                'This method is only available for the test dataset')

        pair_datasets = []

        for ev in self.events:

            subset = self.dataset[self.dataset[:, self.EVENT_IDX] == ev]
            # create a pair dataset for the test dataset. All possible pairs of hits are considered.
            # The label is set to 1 if the two hits belong to the same particle, and 0 otherwise.
            # If the particle id is 0, the label is set to 0.
            n = len(subset)
            hit1, hit2 = np.triu_indices(n, 1)  # get upper triangular indices
            
            total_pairs = len(hit1)
            for start_idx in tqdm(range(0, total_pairs, batch_size)):
                end_idx = min(start_idx + batch_size, total_pairs)
                
                batch_hit1 = hit1[start_idx:end_idx]
                batch_hit2 = hit2[start_idx:end_idx]

                particle_id1 = subset[batch_hit1, self.PID_IDX]
                particle_id2 = subset[batch_hit2, self.PID_IDX]

                labels = np.where((particle_id1 == particle_id2) & (particle_id1 != 0), 1, 0)

                features_hit1 = subset[batch_hit1, :self.PREDICTION_FEATURES]
                features_hit2 = subset[batch_hit2, :self.PREDICTION_FEATURES]

                pair_dataset = np.hstack((features_hit1, features_hit2, labels.reshape(-1, 1))).astype(np.float32)
                np.random.shuffle(pair_dataset)
                pair_datasets.append(pair_dataset)

                del batch_hit1, batch_hit2, particle_id1, particle_id2, labels, features_hit1, features_hit2, pair_dataset
            
            del subset, hit1, hit2

        np.random.shuffle(pair_datasets)
        self.pair_dataset = np.vstack(pair_datasets)

        del pair_datasets

    def save_pair_dataset(self, filename: str):
        '''
            Save the pair dataset to a file
        '''
        print('Saving pair dataset to'+tc.BLUE+tc.UNDERLINE+f'{filename}'+tc.RESET)
        np.save(filename, self.pair_dataset)

    def load_pair_dataset(self, filename: str):
        '''
            Load the pair dataset from a file
        '''
        print('Loading pair dataset from'+tc.BLUE+tc.UNDERLINE+f'{filename}'+tc.RESET)
        self.pair_dataset = np.load(filename)

    def __str__(self):
        return f'BinaryHandler object.\nDataset:\n\t- type: {type(self.dataset)}\n\t- shape: {self.dataset.shape}\n'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        assert 0 <= idx < len(
            self.dataset), f"Index {idx} out of bounds for dataset of size {len(self.dataset)}"
        return self.X[idx], self.y[idx]

    @property
    def X(self):
        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2]

    @property
    def y(self):
        return self.pair_dataset[:, self.PREDICTION_FEATURES*2]

    @property
    def Xmean(self):
        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2].mean(axis=0)

    @property
    def Xstd(self):
        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2].std(axis=0)

    def feature_scaling(self, mean, std):
        '''
            Normalize the first 10 columns of the dataset
        '''
        print('Feature scaling...')
        self.dataset[:, :self.PREDICTION_FEATURES] = (
            self.dataset[:, :self.PREDICTION_FEATURES] - mean[:self.PREDICTION_FEATURES]) / std[:self.PREDICTION_FEATURES]
        if self.pair_dataset is not None:
            self.pair_dataset[:, :self.PREDICTION_FEATURES*2] = (
                self.pair_dataset[:, :self.PREDICTION_FEATURES*2] - mean) / std

    def reduce_to_n_particles(self, n_particles: int):
        '''
            Reduce the dataset to only contain the first n_particles
        '''
        print(f'Reducing dataset to {n_particles} particles...')
        
        particle_ids = self.dataset[:, self.PID_IDX]
        selected_particles = np.unique(particle_ids)[1:n_particles+1]
        # select a noise fraction to keep
        noise_fraction = n_particles/len(np.unique(particle_ids))
        noise = np.where(particle_ids == 0)[0]
        np.random.shuffle(noise)
        noise = noise[:int(noise_fraction*len(noise))]
        selected_particles = np.append(selected_particles, particle_ids[noise])

        self.dataset = self.dataset[np.isin(particle_ids, selected_particles)]
        if self.pair_dataset is not None:
            self.pair_dataset = self.pair_dataset[np.isin(
                particle_ids, selected_particles)]
