'''
    Class for data processing
'''

import sys
sys.path.append('..')
from .vertex import Vertexing
from .geometry import DetectorHashTable
from utils.terminal_colors import TerminalColors as tc

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

from multiprocessing import Pool

import os
import tempfile

from torch.utils.data import Dataset
import torch


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


def calculate_solid_angle(args):
    '''
        Standalone version of pre-existing method in Vertexing class.
        Used for parallel processing.
    '''
    id1, id2, feature_dataset, vertexing_instance = args
    return vertexing_instance.evaluate_solid_angle(feature_dataset[id1, :3], feature_dataset[id2, :3])
#
#
#class BinaryHandler(Dataset):
#
#    _hashtable = None
#
#    def __init__(self, data_dir: str, events: list, save_dir: str, detector_file: str = None, **kwargs):
#        '''
#            Class to create a dataset from the csv files for the binary classification task
#
#            Parameters:
#            - data_dir: directory where the csv files are stored
#            - events: list of event numbers to be processed
#            - type: type of the dataset (train or test)
#
#            pair_dataset: numpy array containing the dataset for the binary classification task
#            dataset: numpy array used for tracking
#                * x: hit coordinate
#                * y: hit coordinate
#                * z: hit coordinate
#                * cluster_size: number of hits in the cluster
#                * amplitude: total energy deposited in the cluster
#                * particle_id: particle id
#                * event_id: event id
#                * hit_id: hit id
#                * unique_module_id: unique module id
#        '''
#
#        self.opt = kwargs.get('opt', 'train')
#        self.mode = kwargs.get('mode', 'load')
#
#        self.save_dir = save_dir
#        self.pair_dataset_file = f'{self.save_dir}/{self.opt}_pair_dataset.npy'
#        self.dataset_file = f'{self.save_dir}/{self.opt}_dataset.npy'
#        self.shapes = {}
#        self.shape_file = f'{self.save_dir}/{self.opt}_shapes.json'
#
#        self.is_lazy = True
#        self.pair_dataset = None
#        self.pair_dataset_chunk_idx = None
#
#        self.events_string = events
#        self.data_dir = data_dir
#        self.events = []
#        self.vertices = []
#
#        self.PREDICTION_FEATURES = 6
#        self.UNIQUE_MODULE_IDX = 5
#        self.PID_IDX = 6
#        self.WEIGHTS_IDX = 7
#        self.EVENT_IDX = 8
#        self.HIT_IDX = 9
#
#        self.NEGATIVE_SAMPLE_RATE = kwargs.get('negative_sample_rate', 2)
#        self.SOLID_ANGLE_THRESHOLD = kwargs.get('solid_angle_threshold', 0.5)
#
#        if BinaryHandler._hashtable is None:
#            BinaryHandler._hashtable = DetectorHashTable(detector_file)
#
#        if self.mode == 'load':
#            print('Dataset is loaded from '+tc.UNDERLINE+tc.CYAN+self.dataset_file+tc.RESET)
#            print('Pair dataset is loaded from '+tc.UNDERLINE+tc.CYAN+self.pair_dataset_file+tc.RESET)
#            with open(self.shape_file, 'r') as f:
#                self.shapes = json.load(f)
#                print(type(self.shapes))
#                print(self.shapes)
#        elif self.mode == 'build':
#            if os.path.exists(self.pair_dataset_file):
#                os.remove(self.pair_dataset_file)
#            if os.path.exists(self.dataset_file):
#                os.remove(self.dataset_file)
#            self._build(kwargs.get('n_particles', None))
#        else:
#            raise ValueError(f'Invalid mode: {self.mode}')
#    
#        
#    def _append_to_memmap(self, data, filename):
#        data = np.atleast_2d(data)
#
#        if not os.path.exists(filename):
#            fp = np.memmap(filename, dtype='float32', mode='w+', shape=data.shape)
#            fp[:] = data[:]
#            self.shapes[filename] = data.shape
#        else:
#            current_size = self.shapes[filename][0]
#            fp_old = np.memmap(filename, dtype='float32', mode='r', shape=(current_size, data.shape[1]))
#            old_data = np.array(fp_old)
#            new_shape = (current_size + data.shape[0], data.shape[1])
#            fp = np.memmap(filename, dtype='float32', mode='w+', shape=new_shape)
#            fp[:current_size] = old_data[:]
#            fp[current_size:] = data[:]
#            self.shapes[filename] = new_shape
#        del fp
#
#        with open(self.shape_file, 'w') as f:
#            json.dump(self.shapes, f)
#
#    def _shuffle_memmap(self, filename, chunk_size: int = int(1e6)):
#        fp = np.memmap(filename, dtype=np.float32, mode='r+', shape=tuple(self.shapes[filename]))
#        num_chunks = len(fp) // chunk_size
#        for i in range(num_chunks):
#            chunk = fp[i*chunk_size:(i+1)*chunk_size]
#            np.random.shuffle(chunk)
#            fp[i*chunk_size:(i+1)*chunk_size] = chunk
#        fp.flush()
#        del fp
#
#    def _shuffle_memmap_on_disk(self, filename, chunk_size: int = int(1e6)):
#        original_shape = tuple(self.shapes[filename])
#        fp = np.memmap(filename, dtype=np.float32, mode='r', shape=original_shape)
#        num_chunks = original_shape[0] // chunk_size
#        indices = np.arange(original_shape[0])
#        np.random.shuffle(indices)
#
#        temp_filename = filename + '.temp'
#        fp_temp = np.memmap(temp_filename, dtype=np.float32, mode='w+', shape=original_shape)
#
#        # Write data to temp file in shuffled order
#        for i in range(num_chunks):
#            chunk_indices = indices[i*chunk_size:(i+1)*chunk_size]
#            fp_temp[chunk_indices] = fp[chunk_indices]
#
#        del fp
#        del fp_temp
#
#        # Read data from temp file in original order and write it back to original file
#        fp = np.memmap(filename, dtype=np.float32, mode='w+', shape=original_shape)
#        fp_temp = np.memmap(temp_filename, dtype=np.float32, mode='r', shape=original_shape)
#
#        for i in range(num_chunks):
#            chunk = fp_temp[i*chunk_size:(i+1)*chunk_size]
#            fp[i*chunk_size:(i+1)*chunk_size] = chunk
#
#        fp.flush()
#        del fp
#        del fp_temp
#
#        # Delete the temporary file
#        os.remove(temp_filename)
#
#    def _build(self, n_particles):  
#        '''
#            Build the dataset
#        '''
#        print(f'Building dataset for {self.opt}...')
#        if self.opt == 'train':
#            self._build_train()
#            if n_particles is not None:
#                self.reduce_to_n_particles(n_particles)
#            self.build_negative_dataset()
#        elif self.opt == 'test':
#            self._build_test()
#            if n_particles is not None:
#                self.reduce_to_n_particles(n_particles)
#            self.build_pair_dataset()
#        else:
#            raise ValueError(f'Invalid dataset type: {self.opt}')
#
#    def _build_train(self):
#        '''
#            Build the training dataset
#        '''
#
#        print('Building training dataset...')
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            # evaluate the vertex
#            vertex_dataset = pd.merge(hits[['hit_id', 'x', 'y', 'z']], truth[[
#                                      'hit_id', 'particle_id']], on='hit_id')
#            vertexing = Vertexing(vertex_dataset)
#            self.vertices.append(vertexing.vertex)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#            particle_ids = truth['particle_id'].unique()
#            # remove noise hits from the particle ids
#            particle_ids = particle_ids[np.where(particle_ids != 0)[0]]
#
#            pair_list = []
#            for p_id in particle_ids:
#                hit_ids = truth[truth['particle_id']
#                                == p_id]['hit_id'].values - 1
#                id1, id2 = np.meshgrid(hit_ids, hit_ids)
#                pairs = np.column_stack((id1.ravel(), id2.ravel()))
#                pair_list.append(pairs[pairs[:, 0] != pairs[:, 1]])
#
#            pairs = np.vstack(pair_list)
#
#            features_idx1 = feature_dataset[pairs[:,
#                                                  0], :self.PREDICTION_FEATURES]
#            features_idx2 = feature_dataset[pairs[:,
#                                                  1], :self.PREDICTION_FEATURES]
#            pair_dataset = np.hstack((features_idx1, features_idx2, np.ones(
#                (features_idx1.shape[0], 1)))).astype(np.float32)
#            np.random.shuffle(pair_dataset)
#
#            self._append_to_memmap(feature_dataset, self.dataset_file)
#            self._append_to_memmap(pair_dataset, self.pair_dataset_file)
#
#            del pair_list, pairs, features_idx1, features_idx2, pair_dataset
#
#    def _build_test(self):
#        '''
#            Build the test dataset
#        '''
#
#        print('Building test dataset...')
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#
#            self._append_to_memmap(feature_dataset, self.dataset_file)
#
#
#    def __str__(self):
#        fp = np.memmap(self.dataset_file, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.dataset_file]))
#        dataset_str = f'BinaryHandler object.\nDataset:\n\t- type: {type(fp)}\n\t- shape: {fp.shape}\n\t- length pairs: {len(self)}\n'
#        del fp
#        return dataset_str
#
#    def __len__(self):
#        return self.shapes[self.pair_dataset_file][0]
#
#    def __getitem__(self, idx):
#        assert 0 <= idx < len(self), f"Index {idx} out of bounds for dataset of size {len(self)}"
#        item = None
#        if self.is_lazy:
#            fp = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.pair_dataset_file]))
#            item = fp[idx].copy()
#            del fp
#        else:
#            # Access only the required chunk to avoid loading all data at once
#            chunk_size = 100000
#            chunk_idx = idx // chunk_size
#            if self.pair_dataset is None or self.pair_dataset_chunk_idx != chunk_idx:
#                start = chunk_idx * chunk_size
#                end = min(start + chunk_size, self.shapes[self.pair_dataset_file][0])
#                fp = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r', shape=(self.shapes[self.pair_dataset_file][0], self.shapes[self.pair_dataset_file][1]))
#                self.pair_dataset = np.array(fp[start:end]).copy()
#                self.pair_dataset_chunk_idx = chunk_idx
#                del fp
#            item = self.pair_dataset[idx % chunk_size]
#        return item[:self.PREDICTION_FEATURES * 2], item[self.PREDICTION_FEATURES * 2]
#
#
#    @property
#    def X(self):
#        fp = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.pair_dataset_file]))
#        X = fp[:, :self.PREDICTION_FEATURES * 2].copy()
#        del fp
#        return X
#
#    @property
#    def y(self):
#        fp = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.pair_dataset_file]))
#        y = fp[:, self.PREDICTION_FEATURES * 2].copy()
#        del fp
#        return y
#
#    @property
#    def Xmean(self):
#        return self.X.mean(axis=0)
#
#    @property
#    def Xstd(self):
#        return self.X.std(axis=0)
#
#    def feature_scaling(self, mean, std, chunk_size=100000):
#        '''
#            Normalize the pair dataset
#        '''
#        print('Feature scaling...')
#        for i in tqdm(range(0, len(self), chunk_size)):
#            fp = np.memmap(self.dataset_file, dtype=np.float32, mode='r+', shape=tuple(self.shapes[self.dataset_file]))
#            fp[i:i + chunk_size, :self.PREDICTION_FEATURES] = (
#                fp[i:i + chunk_size, :self.PREDICTION_FEATURES] - mean[:self.PREDICTION_FEATURES]) / std[:self.PREDICTION_FEATURES]
#            fp.flush()
#            del fp
#        if self.pair_dataset_file is not None:
#            for i in tqdm(range(0, len(self), chunk_size)):
#                fp = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r+', shape=tuple(self.shapes[self.pair_dataset_file]))
#                fp[i:i + chunk_size, :self.PREDICTION_FEATURES * 2] = (
#                    fp[i:i + chunk_size, :self.PREDICTION_FEATURES * 2] - mean) / std
#                fp.flush()
#                del fp
#
#    def reduce_to_n_particles(self, n_particles: int):
#        '''
#            Reduce the dataset to only contain the first n_particles
#        '''
#        print(f'Reducing dataset to {n_particles} particles...')
#
#        fp = np.memmap(self.dataset_file, dtype=np.float32, mode='r+', shape=tuple(self.shapes[self.dataset_file]))
#        particle_ids = fp[:, self.PID_IDX]
#        selected_particles = np.random.choice(
#            np.unique(particle_ids)[1:], n_particles, replace=False)
#        noise_fraction = n_particles / len(np.unique(particle_ids))
#        noise = np.where(particle_ids == 0)[0]
#        noise = noise[:int(noise_fraction * len(noise))]
#        selected_particles = np.append(selected_particles, particle_ids[noise])
#
#        selected_fp = fp[np.isin(particle_ids, selected_particles)]
#
#        # Create a new memory-mapped array for the reduced dataset
#        reduced_fp = np.memmap(self.dataset_file, dtype=np.float32, mode='w+', shape=selected_fp.shape)
#        self.shapes[self.dataset_file] = selected_fp.shape
#        reduced_fp[:] = selected_fp
#        reduced_fp.flush()
#        del fp, reduced_fp
#
#    def build_negative_dataset_solid_angle(self, negative_sample_rate: int = 3):
#
#        print('Building negative dataset...')
#        n_pos_pairs = len(self) / len(self.events_string)
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            # evaluate the vertex
#            vertex_dataset = pd.merge(hits[['hit_id', 'x', 'y', 'z']], truth[[
#                                      'hit_id', 'particle_id']], on='hit_id')
#            vertexing = Vertexing(vertex_dataset)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#
#            # negative samples
#
#            minimum_size = int(n_pos_pairs*self.NEGATIVE_SAMPLE_RATE)
#            current_negative_size = 0
#            while current_negative_size < minimum_size:
#
#                p_id = truth['particle_id'].values
#                id1_list = np.random.randint(
#                    len(feature_dataset), size=minimum_size)
#                id2_list = np.random.randint(
#                    len(feature_dataset), size=minimum_size)
#                pair = np.hstack((id1_list.reshape(minimum_size, 1),
#                                 id2_list.reshape(minimum_size, 1)))
#
#                # use the solid angle to filter the pairs
#                with Pool() as p:
#                    solid_angle = p.map(calculate_solid_angle, [(
#                        id1, id2, feature_dataset, vertexing) for id1, id2 in zip(id1_list, id2_list)])
#
#                solid_angle = np.array(solid_angle)
#
#                # solid_angle = np.zeros(minimum_size)
#                # for i in range(minimum_size):
#                #    solid_angle[i] = vertexing.evaluate_solid_angle(
#                #        feature_dataset[id1_list[i], :3], feature_dataset[id2_list[i], :3])
#
#                accepted_solid_angle_idx = np.where(
#                    solid_angle > self.SOLID_ANGLE_THRESHOLD)[0]
#                rejected_solid_angle_idx = np.where(
#                    solid_angle <= self.SOLID_ANGLE_THRESHOLD)[0]
#                # recover 5% of non accepted solid angle pairs
#                #recovered_rejected_idx = np.random.choice(
#                #    rejected_solid_angle_idx, int(0.05*len(accepted_solid_angle_idx)/0.95))
#                # recover 50% of non accepted solid angle pairs
#                recovered_rejected_idx = np.random.choice(
#                    rejected_solid_angle_idx, int(0.9*len(accepted_solid_angle_idx)/0.1))
#                accepted_idx = np.vstack(
#                    [accepted_solid_angle_idx.reshape(-1, 1), recovered_rejected_idx.reshape(-1, 1)])
#
#                pair = pair[accepted_idx]
#                id1_list = id1_list[accepted_idx]
#                id2_list = id2_list[accepted_idx]
#
#                pair = pair[((p_id[id1_list] == 0) | (
#                    p_id[id1_list] != p_id[id2_list]))]
#
#                features_idx1 = feature_dataset[pair[:,
#                                                     0], :self.PREDICTION_FEATURES]
#                features_idx2 = feature_dataset[pair[:,
#                                                     1], :self.PREDICTION_FEATURES]
#                negative_dataset = np.hstack((features_idx1, features_idx2, np.zeros(
#                    (features_idx1.shape[0], 1)))).astype(np.float32)
#                np.random.shuffle(negative_dataset)
#                current_negative_size += len(negative_dataset)
#                self._append_to_memmap(negative_dataset, self.pair_dataset_file)
#
#                del id1_list, id2_list, pair, solid_angle
#                del features_idx1, features_idx2
#
#            del hits, cells, truth, particles
#            del feature_dataset, negative_dataset
#
#    def build_negative_dataset(self, negative_sample_rate: int = 3):
#
#        print('Building negative dataset...')
#        n_pos_pairs = len(self) / len(self.events_string)
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            # evaluate the vertex
#            vertex_dataset = pd.merge(hits[['hit_id', 'x', 'y', 'z']], truth[[
#                                      'hit_id', 'particle_id']], on='hit_id')
#            vertexing = Vertexing(vertex_dataset)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#
#            # negative samples
#
#            minimum_size = int(n_pos_pairs*self.NEGATIVE_SAMPLE_RATE)
#            
#            p_id = truth['particle_id'].values
#            id1_list = np.random.randint(
#                len(feature_dataset), size=minimum_size)
#            id2_list = np.random.randint(
#                len(feature_dataset), size=minimum_size)
#            pair = np.hstack((id1_list.reshape(minimum_size, 1),
#                             id2_list.reshape(minimum_size, 1)))
#
#            pair = pair[((p_id[id1_list] == 0) | (
#                p_id[id1_list] != p_id[id2_list]))]
#
#            features_idx1 = feature_dataset[pair[:,
#                                                 0], :self.PREDICTION_FEATURES]
#            features_idx2 = feature_dataset[pair[:,
#                                                 1], :self.PREDICTION_FEATURES]
#            negative_dataset = np.hstack((features_idx1, features_idx2, np.zeros(
#                (features_idx1.shape[0], 1)))).astype(np.float32)
#            np.random.shuffle(negative_dataset)
#            self._append_to_memmap(negative_dataset, self.pair_dataset_file)
#
#            del id1_list, id2_list, pair
#            del features_idx1, features_idx2
#
#            del hits, cells, truth, particles
#            del feature_dataset, negative_dataset
#
#        self._shuffle_memmap_on_disk(self.pair_dataset_file)
#
#    def build_pair_dataset(self, batch_size: int = 100000):
#        '''
#           Build the pair dataset for the test dataset.
#        '''
#        
#        print('Building pair dataset...')
#
#        # Create a memory-mapped array for the pair dataset
#        initial_size = 1
#        pair_dataset = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='w+', shape=(initial_size, 2 * self.PREDICTION_FEATURES + 1))
#
#        for ev in self.events_string:
#
#            print(f'Processing event {ev}')
#
#            num_ev = float(int(ev.lstrip('0')))
#            dataset = np.memmap(self.dataset_file, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.dataset_file]))
#            subset = dataset[dataset[:, self.EVENT_IDX] == num_ev]
#
#            n = len(subset)
#            hit1, hit2 = np.triu_indices(n, 1)  # get upper triangular indices
#            total_pairs = len(hit1)
#
#            for start_idx in tqdm(range(0, total_pairs, batch_size)):
#
#                end_idx = min(start_idx + batch_size, total_pairs)
#                batch_hit1 = hit1[start_idx:end_idx]
#                batch_hit2 = hit2[start_idx:end_idx]
#                particle_id1 = subset[batch_hit1, self.PID_IDX]
#                particle_id2 = subset[batch_hit2, self.PID_IDX]
#                labels = np.where((particle_id1 == particle_id2)
#                                  & (particle_id1 != 0), 1, 0)
#
#                features_hit1 = subset[batch_hit1, :self.PREDICTION_FEATURES]
#                features_hit2 = subset[batch_hit2, :self.PREDICTION_FEATURES]
#                batch_pairs = np.hstack(
#                    (features_hit1, features_hit2, labels.reshape(-1, 1))).astype(np.float32)
#                np.random.shuffle(batch_pairs)
#
#                pair_dataset_shape = pair_dataset.shape
#                pair_dataset_dtype = pair_dataset.dtype
#                del pair_dataset
#                with open(self.pair_dataset_file, 'a') as f:
#                    f.truncate((pair_dataset_shape[0] + batch_pairs.shape[0]) * pair_dataset_dtype.itemsize)
#                # Append batch_pairs to the pair_dataset
#                pair_dataset = np.memmap(self.pair_dataset_file, dtype=np.float32, mode='r+', shape=(pair_dataset_shape[0] + batch_pairs.shape[0], 2 * self.PREDICTION_FEATURES + 1))
#                pair_dataset[-batch_pairs.shape[0]:] = batch_pairs
#                initial_size = 0  # Reset initial_size after the first batch
#
#                del batch_hit1, batch_hit2, particle_id1, particle_id2, labels, features_hit1, features_hit2, batch_pairs
#            del subset, hit1, hit2
#
#        self.shapes[self.pair_dataset_file] = pair_dataset.shape
#        with open(self.shape_file, 'w') as f:
#            json.dump(self.shapes, f)
#        print('Pair dataset built.')
#
#    def hard_negative_mining(self, model, device, threshold: float = 0.5, mining_size: int = int(3*1e7), batch_size: int = int(1e6)):
#        '''
#            Perform hard negative mining on the train dataset.
#            In this procedure, negative samples are randomly produced and the model is used to predict the labels.
#            The samples with the highest prediction scores are selected as hard negatives.
#            The model will train on these hard negatives in addition to the positive samples.
#        '''
#        print('Performing hard negative mining...')
#        if self.opt != 'train':
#            raise ValueError(
#                'This method is only available for the test dataset')
#
#        for ev in tqdm(self.events_string):
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#            particle_id_list = truth['particle_id'].values
#
#            for batch_step in range(0, mining_size, batch_size):
#                # create random pairs of hits
#                id1_list = np.random.randint(len(hits), size=batch_size)
#                id2_list = np.random.randint(len(hits), size=batch_size)
#                pairs = np.hstack((id1_list.reshape(batch_size, 1),
#                                   id2_list.reshape(batch_size, 1)))
#                # select only pairs with different particle ids (negative pairs)
#                pairs = pairs[(particle_id_list[id1_list] != particle_id_list[id2_list]) | (
#                    particle_id_list[id1_list] != 0)]
#
#                negative_dataset = np.hstack((feature_dataset[pairs[:, 0], :self.PREDICTION_FEATURES],
#                                             feature_dataset[pairs[:, 1], :self.PREDICTION_FEATURES], np.zeros((len(pairs), 1)))).astype(np.float32)
#
#                model.eval()
#                with torch.no_grad():
#                    predictions = model(torch.tensor(
#                        negative_dataset[:, :self.PREDICTION_FEATURES*2], dtype=torch.float32).to(device)).squeeze(1).detach().float()
#                mask = np.where(predictions > threshold)[0]
#                negative_dataset = negative_dataset[mask]
#
#                self._append_to_memmap(negative_dataset, self.pair_dataset_file)
#
#                del id1_list, id2_list, pairs, negative_dataset, predictions, mask
#
#            del hits, cells, truth, particles
#            del cl_size, amplitude, event_id, particle_id_list
#            del feature_dataset
#
#        self._shuffle_memmap(self.pair_dataset_file)
#
#    def become_lazy(self):
#        self.is_lazy = True
#        del self.pair_dataset
#        self.pair_dataset = None
#
#    def become_eager(self):
#        self.is_lazy = False
#        chunk_size = 100000  # Adjust based on your memory capacity
#        chunks = []
#        with open(self.pair_dataset_file, 'rb') as f:
#            for start in range(0, self.shapes[self.pair_dataset_file][0], chunk_size):
#                end = min(start + chunk_size, self.shapes[self.pair_dataset_file][0])
#                fp = np.memmap(f, dtype=np.float32, mode='r', shape=tuple(self.shapes[self.pair_dataset_file]))
#                chunks.append(np.array(fp[start:end]).copy())
#                del fp
#        self.pair_dataset = np.vstack(chunks)
        


###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################


#class BinaryHandler(Dataset):
#
#    _hashtable = None
#
#    def __init__(self, data_dir: str, events: list, opt: str = 'train', detector_file: str = None, **kwargs):
#        '''
#            Class to create a dataset from the csv files for the binary classification task.
#
#            Parameters:
#            - data_dir: directory where the csv files are stored
#            - events: list of event numbers to be processed
#            - type: type of the dataset (train or test)
#
#            pair_dataset: numpy array containing the dataset for the binary classification task
#            dataset: numpy array used for tracking
#                * x: hit coordinate
#                * y: hit coordinate
#                * z: hit coordinate
#                * cluster_size: number of hits in the cluster
#                * amplitude: total energy deposited in the cluster
#                * particle_id: particle id
#                * event_id: event id
#                * hit_id: hit id
#                * unique_module_id: unique module id
#        '''
#
#        self.pair_dataset = None
#        self.dataset = None
#
#        self.events_string = events
#        self.data_dir = data_dir
#        self.events = []
#        self.vertices = []
#
#        self.opt = opt
#
#        self.PREDICTION_FEATURES = 6
#        self.UNIQUE_MODULE_IDX = 5
#        self.PID_IDX = 6
#        self.WEIGHTS_IDX = 7
#        self.EVENT_IDX = 8
#        self.HIT_IDX = 9
#
#        self.NEGATIVE_SAMPLE_RATE = kwargs.get('negative_sample_rate', 3)
#        self.SOLID_ANGLE_THRESHOLD = kwargs.get('solid_angle_threshold', 0.5)
#
#        if BinaryHandler._hashtable is None:
#            BinaryHandler._hashtable = DetectorHashTable(detector_file)
#
#        if opt == 'train':
#            self._build_train()
#        elif opt == 'test':
#            self._build_test()
#        else:
#            raise ValueError(f'Invalid dataset type: {opt}')
#
#    def _build_train(self):
#        '''
#            Build the training dataset
#        '''
#
#        print('Building training dataset...')
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            # evaluate the vertex
#            vertex_dataset = pd.merge(hits[['hit_id', 'x', 'y', 'z']], truth[[
#                                      'hit_id', 'particle_id']], on='hit_id')
#            vertexing = Vertexing(vertex_dataset)
#            self.vertices.append(vertexing.vertex)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#            particle_ids = truth['particle_id'].unique()
#            # remove noise hits from the particle ids
#            particle_ids = particle_ids[np.where(particle_ids != 0)[0]]
#
#            pair_list = []
#            for p_id in particle_ids:
#                hit_ids = truth[truth['particle_id']
#                                == p_id]['hit_id'].values - 1
#                id1, id2 = np.meshgrid(hit_ids, hit_ids)
#                pairs = np.column_stack((id1.ravel(), id2.ravel()))
#                pair_list.append(pairs[pairs[:, 0] != pairs[:, 1]])
#
#            pairs = np.vstack(pair_list)
#
#            features_idx1 = feature_dataset[pairs[:,
#                                                  0], :self.PREDICTION_FEATURES]
#            features_idx2 = feature_dataset[pairs[:,
#                                                  1], :self.PREDICTION_FEATURES]
#            pair_dataset = np.hstack((features_idx1, features_idx2, np.ones(
#                (features_idx1.shape[0], 1)))).astype(np.float32)
#
#            if self.dataset is None:
#                self.dataset = feature_dataset
#            else:
#                self.dataset = np.vstack((self.dataset, feature_dataset))
#
#            if self.pair_dataset is None:
#                self.pair_dataset = pair_dataset
#            else:
#                self.pair_dataset = np.vstack(
#                    (self.pair_dataset, pair_dataset))
#
#            del pair_list, pairs, features_idx1, features_idx2, pair_dataset
#
#        np.random.shuffle(self.pair_dataset)
#
#    def _build_test(self):
#        '''
#            Build the test dataset
#        '''
#
#        print('Building test dataset...')
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#
#            if self.dataset is None:
#                self.dataset = feature_dataset
#            else:
#                self.dataset = np.vstack((self.dataset, feature_dataset))
#
#            del hits, cells, truth, particles
#            del feature_dataset
#
#    def __str__(self):
#        return f'BinaryHandler object.\nDataset:\n\t- type: {type(self.dataset)}\n\t- shape: {self.dataset.shape}\n'
#
#    def __len__(self):
#        return len(self.pair_dataset)
#
#    def __getitem__(self, idx):
#        assert 0 <= idx < len(
#            self.pair_dataset), f"Index {idx} out of bounds for dataset of size {len(self.pair_dataset)}"
#        return self.X[idx], self.y[idx]
#
#    @property
#    def X(self):
#        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2]
#
#    @property
#    def y(self):
#        return self.pair_dataset[:, self.PREDICTION_FEATURES*2]
#
#    @property
#    def Xmean(self):
#        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2].mean(axis=0)
#
#    @property
#    def Xstd(self):
#        return self.pair_dataset[:, :self.PREDICTION_FEATURES*2].std(axis=0)
#
#    def feature_scaling(self, mean, std, chunk_size=10000):
#        '''
#            Normalize the pair dataset
#        '''
#        print('Feature scaling...')
#        for i in range(0, len(self.dataset), chunk_size):
#            self.dataset[i:i+chunk_size, :self.PREDICTION_FEATURES] = (
#                self.dataset[i:i+chunk_size, :self.PREDICTION_FEATURES] - mean[:self.PREDICTION_FEATURES]) / std[:self.PREDICTION_FEATURES]
#        if self.pair_dataset is not None:
#            for i in range(0, len(self.pair_dataset), chunk_size):
#                self.pair_dataset[i:i+chunk_size, :self.PREDICTION_FEATURES*2] = (
#                    self.pair_dataset[i:i+chunk_size, :self.PREDICTION_FEATURES*2] - mean) / std
#
#    def reduce_to_n_particles(self, n_particles: int):
#        '''
#            Reduce the dataset to only contain the first n_particles
#        '''
#        print(f'Reducing dataset to {n_particles} particles...')
#
#        particle_ids = self.dataset[:, self.PID_IDX]
#        selected_particles = np.random.choice(
#            np.unique(particle_ids)[1:], n_particles, replace=False)
#        # select a noise fraction to keep
#        noise_fraction = n_particles/len(np.unique(particle_ids))
#        noise = np.where(particle_ids == 0)[0]
#        noise = noise[:int(noise_fraction*len(noise))]
#        selected_particles = np.append(selected_particles, particle_ids[noise])
#
#        self.dataset = self.dataset[np.isin(particle_ids, selected_particles)]
#        if self.pair_dataset is not None:
#            self.pair_dataset = self.pair_dataset[np.isin(
#                particle_ids, selected_particles)]
#
#    def build_negative_dataset(self, negative_sample_rate: int = 3):
#
#        print('Building negative dataset...')
#
#        negative_datasets = []
#
#        for ev in self.events_string:
#            print(f'Processing event {ev}')
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            # evaluate the vertex
#            vertex_dataset = pd.merge(hits[['hit_id', 'x', 'y', 'z']], truth[[
#                                      'hit_id', 'particle_id']], on='hit_id')
#            vertexing = Vertexing(vertex_dataset)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#
#            n_pos_pairs = len(self.pair_dataset) / len(self.events_string)
#
#            # negative samples
#
#            minimum_size = int(n_pos_pairs*self.NEGATIVE_SAMPLE_RATE)
#            current_negative_size = 0
#            while current_negative_size < minimum_size:
#
#                p_id = truth['particle_id'].values
#                id1_list = np.random.randint(
#                    len(feature_dataset), size=minimum_size)
#                id2_list = np.random.randint(
#                    len(feature_dataset), size=minimum_size)
#                pair = np.hstack((id1_list.reshape(minimum_size, 1),
#                                 id2_list.reshape(minimum_size, 1)))
#
#                # use the solid angle to filter the pairs
#                with Pool() as p:
#                    solid_angle = p.map(calculate_solid_angle, [(
#                        id1, id2, feature_dataset, vertexing) for id1, id2 in zip(id1_list, id2_list)])
#
#                solid_angle = np.array(solid_angle)
#
#                # solid_angle = np.zeros(minimum_size)
#                # for i in range(minimum_size):
#                #    solid_angle[i] = vertexing.evaluate_solid_angle(
#                #        feature_dataset[id1_list[i], :3], feature_dataset[id2_list[i], :3])
#
#                accepted_solid_angle_idx = np.where(
#                    solid_angle > self.SOLID_ANGLE_THRESHOLD)[0]
#                rejected_solid_angle_idx = np.where(
#                    solid_angle <= self.SOLID_ANGLE_THRESHOLD)[0]
#                # recover 5% of non accepted solid angle pairs
#                recovered_rejected_idx = np.random.choice(
#                    rejected_solid_angle_idx, int(0.75*len(accepted_solid_angle_idx)/0.25))
#                accepted_idx = np.vstack(
#                    [accepted_solid_angle_idx.reshape(-1, 1), recovered_rejected_idx.reshape(-1, 1)])
#
#                pair = pair[accepted_idx]
#                id1_list = id1_list[accepted_idx]
#                id2_list = id2_list[accepted_idx]
#
#                pair = pair[((p_id[id1_list] == 0) | (
#                    p_id[id1_list] != p_id[id2_list]))]
#
#                features_idx1 = feature_dataset[pair[:,
#                                                     0], :self.PREDICTION_FEATURES]
#                features_idx2 = feature_dataset[pair[:,
#                                                     1], :self.PREDICTION_FEATURES]
#                negative_dataset = np.hstack((features_idx1, features_idx2, np.zeros(
#                    (features_idx1.shape[0], 1)))).astype(np.float32)
#                current_negative_size += len(negative_dataset)
#                negative_datasets.append(negative_dataset)
#
#                del id1_list, id2_list, pair, solid_angle
#                del features_idx1, features_idx2
#
#            del hits, cells, truth, particles
#            del feature_dataset, negative_dataset
#
#        negative_dataset = np.concatenate(negative_datasets)
#        self.pair_dataset = np.concatenate(
#            (self.pair_dataset, negative_dataset), dtype=np.float32)
#
#        del negative_datasets, negative_dataset
#        np.random.shuffle(self.pair_dataset)
#
#    def build_pair_dataset(self, batch_size: int = 10000):
#        '''
#           Build the pair dataset for the test dataset.
#        '''
#
#        print('Building pair dataset...')
#
#        temp_files = []
#
#        for ev in self.events_string:
#
#            print(f'Processing event {ev}')
#
#            num_ev = float(int(ev.lstrip('0')))
#            subset = self.dataset[self.dataset[:, self.EVENT_IDX] == num_ev]
#            n = len(subset)
#            hit1, hit2 = np.triu_indices(n, 1)  # get upper triangular indices
#
#            total_pairs = len(hit1)
#            for start_idx in tqdm(range(0, total_pairs, batch_size)):
#                end_idx = min(start_idx + batch_size, total_pairs)
#
#                batch_hit1 = hit1[start_idx:end_idx]
#                batch_hit2 = hit2[start_idx:end_idx]
#
#                particle_id1 = subset[batch_hit1, self.PID_IDX]
#                particle_id2 = subset[batch_hit2, self.PID_IDX]
#
#                labels = np.where((particle_id1 == particle_id2)
#                                  & (particle_id1 != 0), 1, 0)
#
#                features_hit1 = subset[batch_hit1, :self.PREDICTION_FEATURES]
#                features_hit2 = subset[batch_hit2, :self.PREDICTION_FEATURES]
#
#                pair_dataset = np.hstack(
#                    (features_hit1, features_hit2, labels.reshape(-1, 1))).astype(np.float32)
#                np.random.shuffle(pair_dataset)
#
#                # Save pair_dataset to a temporary file
#                temp_file = tempfile.NamedTemporaryFile(delete=False)
#                print('Saving temporary file '+tc.UNDERLINE +
#                      tc.CYAN+f'{temp_file.name}'+tc.RESET)
#                np.save(temp_file, pair_dataset)
#                temp_files.append(temp_file.name)
#
#                del batch_hit1, batch_hit2, particle_id1, particle_id2, labels, features_hit1, features_hit2, pair_dataset
#
#            del subset, hit1, hit2
#
#        # Load and concatenate pair datasets from temporary files
#        self.pair_dataset = np.concatenate(
#            [np.load(file) for file in temp_files])
#
#        # Delete temporary files
#        for file in temp_files:
#            os.remove(file)
#
#        print('Pair dataset built.')
#
#    def save_train_dataset(self, filename: str):
#        '''
#            Save the train dataset to a file
#        '''
#        print('Saving train dataset to '+tc.BLUE +
#              tc.UNDERLINE+f'{filename}'+tc.RESET)
#
#        np.save(f'{filename}', self.pair_dataset)
#
#    def load_train_dataset(self, filename: str):
#        '''
#            Load the train dataset from a file
#        '''
#        print('Loading train dataset from '+tc.BLUE +
#              tc.UNDERLINE+f'{filename}'+tc.RESET)
#
#        self.pair_dataset = np.load(filename)
#
#    def save_pair_dataset(self, filename: str):
#        '''
#            Save the pair dataset to a file
#        '''
#        print('Saving pair dataset to '+tc.BLUE +
#              tc.UNDERLINE+f'{filename}'+tc.RESET)
#
#        # Adjust this value based on your available memory
#        chunk_size = int(1e7)
#        for i in range(0, len(self.pair_dataset), chunk_size):
#            np.save(f'{filename}_chunk{i//chunk_size}',
#                    self.pair_dataset[i:i+chunk_size])
#
#    def load_pair_dataset(self, filename: str):
#        '''
#            Load the pair dataset from a file
#        '''
#        print('Loading pair dataset from '+tc.BLUE +
#              tc.UNDERLINE+f'{filename}'+tc.RESET)
#
#        # Get a list of all files that start with the filename
#        files = glob.glob(f'{filename}_chunk*')
#
#        # Load each file and append it to a list
#        chunks = [np.load(file) for file in files]
#
#        # Concatenate all chunks to form the complete dataset
#        self.pair_dataset = np.concatenate(chunks, axis=0)
#
#    def hard_negative_mining(self, model, device, threshold: float = 0.5, mining_size: int = int(3*1e7), batch_size: int = int(1e6)):
#        '''
#            Perform hard negative mining on the train dataset.
#            In this procedure, negative samples are randomly produced and the model is used to predict the labels.
#            The samples with the highest prediction scores are selected as hard negatives.
#            The model will train on these hard negatives in addition to the positive samples.
#        '''
#        print('Performing hard negative mining...')
#        if self.opt != 'train':
#            raise ValueError(
#                'This method is only available for the test dataset')
#
#        negative_mining = []
#
#        for ev in tqdm(self.events_string):
#
#            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
#            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
#            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
#            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')
#
#            # store cluster size in a numpy array
#            cl_size = cells.groupby(['hit_id'])['value'].count().values
#            # store total energy deposited in a numpy array
#            amplitude = cells.groupby(['hit_id'])['value'].sum().values
#
#            num_ev = float(int(ev.lstrip('0')))
#            event_id = np.ones((len(hits), 1))*int(num_ev)
#            self.events.append(num_ev)
#
#            # add unique module id to the hits
#            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)
#
#            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
#                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
#            particle_id_list = truth['particle_id'].values
#
#            for batch_step in range(0, mining_size, batch_size):
#                # create random pairs of hits
#                id1_list = np.random.randint(len(hits), size=batch_size)
#                id2_list = np.random.randint(len(hits), size=batch_size)
#                pairs = np.hstack((id1_list.reshape(batch_size, 1),
#                                   id2_list.reshape(batch_size, 1)))
#                # select only pairs with different particle ids (negative pairs)
#                pairs = pairs[(particle_id_list[id1_list] != particle_id_list[id2_list]) | (
#                    particle_id_list[id1_list] != 0)]
#
#                negative_dataset = np.hstack((feature_dataset[pairs[:, 0], :self.PREDICTION_FEATURES],
#                                             feature_dataset[pairs[:, 1], :self.PREDICTION_FEATURES], np.zeros((len(pairs), 1)))).astype(np.float32)
#
#                model.eval()
#                with torch.no_grad():
#                    predictions = model(torch.tensor(
#                        negative_dataset[:, :self.PREDICTION_FEATURES*2], dtype=torch.float32).to(device)).squeeze(1).detach().float()
#                mask = np.where(predictions > threshold)[0]
#                negative_dataset = negative_dataset[mask]
#
#                if len(negative_mining) == 0:
#                    negative_mining = negative_dataset
#                else:
#                    negative_mining = np.vstack(
#                        (negative_mining, negative_dataset))
#
#                del id1_list, id2_list, pairs, negative_dataset, predictions, mask
#
#            del hits, cells, truth, particles
#            del cl_size, amplitude, event_id, particle_id_list
#            del feature_dataset,
#
#        self.pair_dataset = np.vstack((self.pair_dataset, negative_mining))

###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################
###########################################################################################################



class BinaryHandler(Dataset):

    _hashtable = None

    def __init__(self, data_dir: str, events: list, opt: str = 'train', detector_file: str = None):
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
                * hit_id: hit id
                * unique_module_id: unique module id
        '''

        self.pair_dataset = None
        self.dataset = None

        self.events_string = events
        self.data_dir = data_dir
        self.events = []

        self.opt = opt

        self.PREDICTION_FEATURES = 6
        self.UNIQUE_MODULE_IDX = 5
        self.PID_IDX = 6
        self.WEIGHTS_IDX = 7
        self.EVENT_IDX = 8
        self.HIT_IDX = 9

        if BinaryHandler._hashtable is None:
            BinaryHandler._hashtable = DetectorHashTable(detector_file)

        if opt == 'train':
            self._build_train()
        elif opt == 'test':
            self._build_test()
        else:
            raise ValueError(f'Invalid dataset type: {opt}')

    def _build_train(self):
        '''
            Build the training dataset
        '''

        print('Building training dataset...')

        for ev in self.events_string:
            print(f'Processing event {ev}')

            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')

            # store cluster size in a numpy array
            cl_size = cells.groupby(['hit_id'])['value'].count().values
            # store total energy deposited in a numpy array
            amplitude = cells.groupby(['hit_id'])['value'].sum().values

            num_ev = float(int(ev.lstrip('0')))
            event_id = np.ones((len(hits), 1))*int(num_ev)
            self.events.append(num_ev)

            # add unique module id to the hits
            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)

            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
            particle_ids = truth['particle_id'].unique()
            # remove noise hits from the particle ids
            particle_ids = particle_ids[np.where(particle_ids != 0)[0]]

            pair_list = []
            for p_id in particle_ids:
                hit_ids = truth[truth['particle_id']
                                == p_id]['hit_id'].values - 1
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
                
            n_pos_pairs = len(pair_dataset)
            del pair_list, pairs, features_idx1, features_idx2, pair_dataset

            # negative samples

            NEGATIVE_SAMPLE_RATE = 3
            size = int(n_pos_pairs*NEGATIVE_SAMPLE_RATE)
            p_id = truth['particle_id'].values
            id1_list = np.random.randint(len(feature_dataset), size=size)
            id2_list = np.random.randint(len(feature_dataset), size=size)
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

    def _build_test(self):
        '''
            Build the test dataset
        '''

        print('Building test dataset...')

        for ev in self.events_string:
            print(f'Processing event {ev}')

            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')

            # store cluster size in a numpy array
            cl_size = cells.groupby(['hit_id'])['value'].count().values
            # store total energy deposited in a numpy array
            amplitude = cells.groupby(['hit_id'])['value'].sum().values

            num_ev = float(int(ev.lstrip('0')))
            event_id = np.ones((len(hits), 1))*int(num_ev)
            self.events.append(num_ev)

            # add unique module id to the hits
            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)

            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)

            if self.dataset is None:
                self.dataset = feature_dataset
            else:
                self.dataset = np.vstack((self.dataset, feature_dataset))

            del hits, cells, truth, particles
            del feature_dataset

    def __str__(self):
        return f'BinaryHandler object.\nDataset:\n\t- type: {type(self.dataset)}\n\t- shape: {self.dataset.shape}\n'

    def __len__(self):
        return len(self.pair_dataset)

    def __getitem__(self, idx):
        assert 0 <= idx < len(
            self.pair_dataset), f"Index {idx} out of bounds for dataset of size {len(self.pair_dataset)}"
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
        noise = noise[:int(noise_fraction*len(noise))]
        selected_particles = np.append(selected_particles, particle_ids[noise])

        self.dataset = self.dataset[np.isin(particle_ids, selected_particles)]
        if self.pair_dataset is not None:
            self.pair_dataset = self.pair_dataset[np.isin(
                particle_ids, selected_particles)]
    
    def build_pair_dataset(self, batch_size: int = 10000):
       '''
          Build the pair dataset for the test dataset.
       '''
       print('Building pair dataset...')
       if self.opt != 'test':
           raise ValueError(
               'This method is only available for the test dataset')
       pair_datasets = []
       #temp_file_name = tempfile.NamedTemporaryFile(delete=False).name + '.npz'
       #print('Temporary file:'+tc.UNDERLINE+tc.CYAN+f'{temp_file_name}'+tc.RESET)
       for ev in self.events_string:
           num_ev = float(int(ev.lstrip('0')))
           subset = self.dataset[self.dataset[:, self.EVENT_IDX] == num_ev]
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
               labels = np.where((particle_id1 == particle_id2)
                                 & (particle_id1 != 0), 1, 0)
               features_hit1 = subset[batch_hit1, :self.PREDICTION_FEATURES]
               features_hit2 = subset[batch_hit2, :self.PREDICTION_FEATURES]
               pair_dataset = np.hstack(
                   (features_hit1, features_hit2, labels.reshape(-1, 1))).astype(np.float32)
               np.random.shuffle(pair_dataset)
               #np.savez_compressed(temp_file_name, pair_dataset)
               pair_datasets.append(pair_dataset)
               del batch_hit1, batch_hit2, particle_id1, particle_id2, labels, features_hit1, features_hit2, pair_dataset
           del subset, hit1, hit2
       self.pair_dataset = np.vstack(pair_datasets)
       #self.pair_dataset = np.load(temp_file_name)['arr_0']
       #os.remove(temp_file_name)
       del pair_datasets

    def save_pair_dataset(self, filename: str):
        '''
            Save the pair dataset to a file
        '''
        print('Saving pair dataset to '+tc.BLUE +
              tc.UNDERLINE+f'{filename}'+tc.RESET)
        np.save(filename, self.pair_dataset)

    def load_pair_dataset(self, filename: str):
        '''
            Load the pair dataset from a file
        '''
        print('Loading pair dataset from '+tc.BLUE +
              tc.UNDERLINE+f'{filename}'+tc.RESET)
        self.pair_dataset = np.load(filename)

    def hard_negative_mining(self, model, device, threshold: float = 0.5, mining_size: int = int(3*1e7), batch_size: int = int(1e6)):
        '''
            Perform hard negative mining on the train dataset.
            In this procedure, negative samples are randomly produced and the model is used to predict the labels.
            The samples with the highest prediction scores are selected as hard negatives.
            The model will train on these hard negatives in addition to the positive samples.
        '''
        print('Performing hard negative mining...')
        if self.opt != 'train':
            raise ValueError(
                'This method is only available for the test dataset')

        negative_mining = []

        for ev in tqdm(self.events_string):

            hits = pd.read_csv(f'{self.data_dir}/event{ev}-hits.csv')
            cells = pd.read_csv(f'{self.data_dir}/event{ev}-cells.csv')
            truth = pd.read_csv(f'{self.data_dir}/event{ev}-truth.csv')
            particles = pd.read_csv(f'{self.data_dir}/event{ev}-particles.csv')

            # store cluster size in a numpy array
            cl_size = cells.groupby(['hit_id'])['value'].count().values
            # store total energy deposited in a numpy array
            amplitude = cells.groupby(['hit_id'])['value'].sum().values

            num_ev = float(int(ev.lstrip('0')))
            event_id = np.ones((len(hits), 1))*int(num_ev)
            self.events.append(num_ev)

            # add unique module id to the hits
            hits = BinaryHandler._hashtable.add_unique_module_id_column(hits)

            feature_dataset = np.hstack((hits[['x', 'y', 'z']], cl_size.reshape(len(cl_size), 1), amplitude.reshape(
                len(amplitude), 1), hits[['unique_module_id']], truth[['particle_id', 'weight']], event_id.reshape(len(event_id), 1), hits[['hit_id']]), dtype=np.float32)
            particle_id_list = truth['particle_id'].values

            for batch_step in range(0, mining_size, batch_size):
                # create random pairs of hits
                id1_list = np.random.randint(len(hits), size=batch_size)
                id2_list = np.random.randint(len(hits), size=batch_size)
                pairs = np.hstack((id1_list.reshape(batch_size, 1),
                                   id2_list.reshape(batch_size, 1)))
                # select only pairs with different particle ids (negative pairs)
                pairs = pairs[(particle_id_list[id1_list] != particle_id_list[id2_list]) | (
                    particle_id_list[id1_list] != 0)]

                negative_dataset = np.hstack((feature_dataset[pairs[:, 0], :self.PREDICTION_FEATURES],
                                             feature_dataset[pairs[:, 1], :self.PREDICTION_FEATURES], np.zeros((len(pairs), 1)))).astype(np.float32)

                model.eval()
                with torch.no_grad():
                    predictions = model(torch.tensor(
                        negative_dataset[:, :self.PREDICTION_FEATURES*2], dtype=torch.float32).to(device)).squeeze(1).detach().float()
                mask = np.where(predictions > threshold)[0]
                negative_dataset = negative_dataset[mask]

                if len(negative_mining) == 0:
                    negative_mining = negative_dataset
                else:
                    negative_mining = np.vstack(
                        (negative_mining, negative_dataset))
                
                del id1_list, id2_list, pairs, negative_dataset, predictions, mask

            del hits, cells, truth, particles
            del cl_size, amplitude, event_id, particle_id_list
            del feature_dataset,
            

        self.pair_dataset = np.vstack((self.pair_dataset, negative_mining))