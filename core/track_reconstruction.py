'''
    Class for track reconstruction
'''

from utils.terminal_colors import TerminalColors as tc
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from multiprocessing import Pool

import torch

from core.data_handler import BinaryHandler

import sys
sys.path.append('..')


class TrackReconstruction:

    def __init__(self, data_handler: BinaryHandler, model, device):
        '''
            Class to reconstruct the track from the hits

            Parameters:
            data_handler (DataHandler): DataHandler object
            model: machine learning model 
        '''

        self.data_handler = data_handler
        self.model = model
        self.device = device

        self.prediction_matrix = None
        self.all_tracks = None
        self.all_tracks_idx = None

        self.tracks = None
        self.track_ids = None
        self.track_dataframe = None

    def compute_prediction_matrix(self, minimum_threshold: float = 0.5, output_file: str = None):
        '''
            Compute an array of lists of hits that have probability to belong to the same track 
            higher than a minimum threshold
        '''
        print('Computing prediction matrix...')

        self.model.eval()

        self.prediction_matrix = []
        dataset = self.data_handler.dataset
        PREDICTION_FEATURES = self.data_handler.PREDICTION_FEATURES

        pairs1 = np.zeros((len(dataset), PREDICTION_FEATURES*2))
        pairs1[:, PREDICTION_FEATURES:] = dataset[:,
                                                       :PREDICTION_FEATURES]
        pairs2 = np.zeros((len(dataset), PREDICTION_FEATURES*2))
        pairs2[:, :PREDICTION_FEATURES] = dataset[:,
                                                       :PREDICTION_FEATURES]

        for ihit in tqdm(range(len(dataset)-1)):

            pairs1[ihit+1:, :PREDICTION_FEATURES] = np.tile(
                dataset[ihit, :PREDICTION_FEATURES], (len(pairs1)-ihit-1, 1))
            predictions1 = self.model(torch.tensor(
                pairs1[ihit+1:], dtype=torch.float32).to(self.device)).squeeze(1).detach().float()
            over_threshold_idx1 = np.where(
                predictions1 > minimum_threshold/2.)[0]

            predictions = np.zeros(len(predictions1))

            if len(over_threshold_idx1) > 0:
                pairs2[over_threshold_idx1+ihit+1, PREDICTION_FEATURES:
                       ] = pairs1[over_threshold_idx1+ihit+1, :PREDICTION_FEATURES]
                predictions2 = self.model(torch.tensor(
                    pairs2[over_threshold_idx1+ihit+1], dtype=torch.float32).to(self.device)).squeeze(1).detach().float()

                predictions[over_threshold_idx1] = (
                    predictions1[over_threshold_idx1] + predictions2) / 2.
                del predictions2

            over_threshold_idx = np.where(predictions > minimum_threshold)[0]
            self.prediction_matrix.append(
                [over_threshold_idx+ihit+1, predictions[over_threshold_idx]])
            
            del predictions1, predictions

        self.prediction_matrix.append(
            [np.array([], dtype='int64'), np.array([], dtype='float32')])
        
        # copilot
        # complete the prediction matrix with the symmetric predictions
        insertions = [[] for _ in range(len(self.prediction_matrix))]

        for ihit in tqdm(range(len(self.prediction_matrix))):
            iihit = len(self.prediction_matrix) - ihit - 1
            for jhit in range(len(self.prediction_matrix[iihit][0])):
                jjhit = self.prediction_matrix[iihit][0][jhit]
                insertions[jjhit].append((iihit, self.prediction_matrix[iihit][1][jhit]))

        for jjhit, inserts in enumerate(insertions):
            if inserts:
                iihits, values = zip(*inserts)
                self.prediction_matrix[jjhit][0] = np.insert(self.prediction_matrix[jjhit][0], 0, iihits)
                self.prediction_matrix[jjhit][1] = np.insert(self.prediction_matrix[jjhit][1], 0, values)

        # complete the prediction matrix with the simmetric predictions
        #for ihit in tqdm(range(len(self.prediction_matrix))):
        #    iihit = len(self.prediction_matrix) - ihit - 1
        #    for jhit in range(len(self.prediction_matrix[iihit][0])):
        #        jjhit = self.prediction_matrix[iihit][0][jhit]
        #        self.prediction_matrix[jjhit][0] = np.insert(
        #            self.prediction_matrix[jjhit][0], 0, iihit)
        #        self.prediction_matrix[jjhit][1] = np.insert(
        #            self.prediction_matrix[jjhit][1], 0, self.prediction_matrix[iihit][1][jhit])
                
        if output_file is not None:
            with open(output_file, 'wb') as f:
                pickle.dump(self.prediction_matrix, f)
            print('Prediction matrix saved to'+tc.CYAN +
                  tc.UNDERLINE+f'{output_file}'+tc.RESET)
            
        del pairs1, pairs2, insertions
            
    def load_prediction_matrix(self, input_file: str):
        '''
            Load the prediction matrix from a file
        '''
        print('Loading prediction matrix from ' +
              tc.CYAN+tc.UNDERLINE+f'{input_file}'+tc.RESET)
        with open(input_file, 'rb') as f:
            self.prediction_matrix = pickle.load(f)

    def track_proposal(self, ihit: int, mask, threshold: float = 0.90, updating_threshold: bool = False):
        '''
            Propose a track starting from a hit. For he first proposal,
            use the probabilities to select only hits above a threshold and then select the one with highest probability.
            Then loop over using the probability for the newly added hit to select the next one.

            For subsequent track proposals, an already computed mask can be used.

            Parameters:
            ihit (int): index of the hit to start the track
            mask (np.array): mask of the hits that are already in a track
            threshold (float): threshold for the model output

            Returns:
            List of hits that belong to the track
        '''

        dataset = self.data_handler.dataset
        HIT_IDX = self.data_handler.HIT_IDX
        UNIQUE_MODULE_IDX = self.data_handler.UNIQUE_MODULE_IDX
        first_hit_id = int(dataset[ihit, HIT_IDX])
        
        trk_idx = [ihit]        # save hit index for loops
        track = [first_hit_id]  # save hit id
        cumulative_prob = 0

        while True:

            current_hit = trk_idx[-1]

            prob_hit = np.zeros(len(self.prediction_matrix))
            prob_hit[self.prediction_matrix[current_hit][0]
                     ] = self.prediction_matrix[current_hit][1]

            if updating_threshold:
                if (len(track) % 5 == 0) and len(track) != 5:
                    threshold = threshold + 0.03

            # automatically remove hits from the same module
            candidate = np.where(prob_hit > threshold)[0]
            if len(candidate) > 0:
                mask[candidate[np.isin(dataset[candidate, UNIQUE_MODULE_IDX], dataset[trk_idx, UNIQUE_MODULE_IDX])]] = 0

            mask = (prob_hit > threshold) * mask
            mask[current_hit] = 0

            cumulative_prob = (prob_hit + cumulative_prob) * mask
            if cumulative_prob.max() < len(track) * threshold:
                break

            next_hit = cumulative_prob.argmax()
            next_hit_id = dataset[next_hit, HIT_IDX]
            if next_hit_id not in track:  # Check if the hit is already in the track
                trk_idx.append(next_hit)
                track.append(int(next_hit_id))

        return track, trk_idx
    
    def process_hit(self, ihit, threshold: float = 0.85):
        '''
            Track reconstruction for a single hit.
        '''

        mask = np.ones(len(self.prediction_matrix))
        trk_prop, trk_prop_idx = self.track_proposal(ihit, mask, threshold)

        if len(trk_prop) > 1:
            # for a better tracking, check if the track radically changes removing
            # th most probable hit
            mask[trk_prop_idx[1]] == 0
            new_trk_prop, new_trk_prop_idx = self.track_proposal(
                ihit, mask, threshold)

            if len(new_trk_prop) > len(trk_prop):
                # if the new track is better, iterate again
                trk_prop = new_trk_prop
                trk_prop_idx = new_trk_prop_idx
                mask[trk_prop_idx[1]] == 0
                new_trk_prop = self.track_proposal(
                    ihit, mask, threshold)
                
                if len(new_trk_prop) > len(trk_prop):
                    # if the new track is better, keep it
                    trk_prop = new_trk_prop
                    trk_prop_idx = new_trk_prop_idx

            elif len(new_trk_prop) > 1:
                # if the new track is worse, but still a track, try removing the second best hit
                mask[trk_prop_idx[1]] == 1
                mask[new_trk_prop_idx[1]] == 0
                new_trk_prop, new_trk_prop_idx = self.track_proposal(
                    ihit, mask, threshold)

                if len(new_trk_prop) > len(trk_prop):
                    # if the new track is better, keep it
                    trk_prop = new_trk_prop
                    trk_prop_idx = new_trk_prop_idx
        
        return trk_prop, trk_prop_idx




    def reconstruct_all_tracks(self, threshold: float = 0.85, output_file: str = None):
        '''
            Reconstruct all tracks from the hits.
            A track is associated to each hit, as a list of the hits that belong to the track.

            Parameters:
            threshold (float): threshold for the model output

            Returns:
            List of tracks
        '''

        print('Reconstructing all track candidates')
        self.all_tracks = []
        self.all_tracks_idx = []

        with Pool() as p:
            all_tracks_array = p.map(self.process_hit, range(len(self.prediction_matrix)))

        self.all_tracks, self.all_tracks_idx = zip(*all_tracks_array)
        
        if output_file is not None:
            with open(output_file, 'wb') as f:
                pickle.dump(all_tracks_array, f)
            print('All tracks saved to'+tc.CYAN +
                  tc.UNDERLINE+f'{output_file}'+tc.RESET)

    def load_all_tracks(self, input_file: str):
        '''
            Load the tracks from a file
        '''
        print('Loading all tracks from '+tc.CYAN +
              tc.UNDERLINE+f'{input_file}'+tc.RESET)
        with open(input_file, 'rb') as f:
            all_tracks_array = pickle.load(f)

        self.all_tracks, self.all_tracks_idx = zip(*all_tracks_array)

    def score_tracks(self, penalty_factor: float = 8.):
        '''
            Score the tracks. The selected score metrics evaluates the quality of the track reconstruction
            by comparing the different reconstructed tracks. For each hit in the track, the difference between
            the shared hits and unshared hits is considered. Shared hits are common hits for the current track and 
            the track associated to the currently considered hit.
        '''

        print('Scoring tracks')
        self.score = []

        for itrack in tqdm(range(len(self.all_tracks))):

            track = self.all_tracks[itrack]
            track_idx = self.all_tracks_idx[itrack]
            track_length = len(track)

            if track_length <= 1:
                self.score.append(-np.inf)
                continue

            shared_hits = 0
            unshared_hits = 0

            for hit_idx, hit in zip(track_idx[1:], track[1:]):
                shared_hits += np.sum(
                    np.isin(self.all_tracks[hit_idx], track, assume_unique=True))
                unshared_hits += len(
                    np.isin(self.all_tracks[hit_idx], track, assume_unique=True, invert=True))

            score = (shared_hits - unshared_hits * penalty_factor -
                     track_length) / ((track_length - 1) * track_length)

            self.score.append(score)

    def select_tracks(self):
        '''
            Select the tracks that have the best score and assing a track_id to each hit
        '''

        print('Selecting tracks')

        self.track_ids = np.zeros(
            len(self.data_handler.dataset), dtype='int64')
        track_id = 0

        # assign track_id from best to worst
        for itrack in tqdm(np.argsort(self.score)[::-1]):

            track = np.array(self.all_tracks_idx[itrack])
            # remove hits that are already assigned to a track
            track = track[np.where(self.track_ids[track] == 0)[0]]

            if len(track) > 3:
                track_id += 1
                self.track_ids[track] = track_id

    def _analyze_tracks(self, truth, submission):
        """Compute the majority particle, hit counts, and weight for each track.

        Parameters
        ----------
        truth : pandas.DataFrame
            Truth information. Must have hit_id, particle_id, and weight columns.
        submission : pandas.DataFrame
            Proposed hit/track association. Must have hit_id and track_id columns.

        Returns
        -------
        pandas.DataFrame
            Contains track_id, nhits, major_particle_id, major_particle_nhits,
            major_nhits, and major_weight columns.
        """

        # true number of hits for each particle_id
        particles_nhits = truth['particle_id'].value_counts(sort=False)
        total_weight = truth['weight'].sum()
        # combined event with minimal reconstructed and truth information
        event = pd.merge(truth[['hit_id', 'particle_id', 'weight']],
                             submission[['hit_id', 'track_id']],
                             on=['hit_id'], how='left', validate='one_to_one')
        event.drop('hit_id', axis=1, inplace=True)
        event.sort_values(by=['track_id', 'particle_id'], inplace=True)

        # ASSUMPTIONs: 0 <= track_id, 0 <= particle_id

        tracks = []
        # running sum for the reconstructed track we are currently in
        rec_track_id = -1
        rec_nhits = 0
        # running sum for the particle we are currently in (in this track_id)
        cur_particle_id = -1
        cur_nhits = 0
        cur_weight = 0
        # majority particle with most hits up to now (in this track_id)
        maj_particle_id = -1
        maj_nhits = 0
        maj_weight = 0

        for hit in tqdm(event.itertuples(index=False)):
            # we reached the next track so we need to finish the current one
            if (rec_track_id != -1) and (rec_track_id != hit.track_id):
                # could be that the current particle is the majority one
                if maj_nhits < cur_nhits:
                    maj_particle_id = cur_particle_id
                    maj_nhits = cur_nhits
                    maj_weight = cur_weight
                # store values for this track
                tracks.append((rec_track_id, rec_nhits, maj_particle_id,
                    particles_nhits[maj_particle_id], maj_nhits,
                    maj_weight / total_weight))

            # setup running values for next track (or first)
            if rec_track_id != hit.track_id:
                rec_track_id = hit.track_id
                rec_nhits = 1
                cur_particle_id = hit.particle_id
                cur_nhits = 1
                cur_weight = hit.weight
                maj_particle_id = -1
                maj_nhits = 0
                maj_weights = 0
                continue

            # hit is part of the current reconstructed track
            rec_nhits += 1

            # reached new particle within the same reconstructed track
            if cur_particle_id != hit.particle_id:
                # check if last particle has more hits than the majority one
                # if yes, set the last particle as the new majority particle
                if maj_nhits < cur_nhits:
                    maj_particle_id = cur_particle_id
                    maj_nhits = cur_nhits
                    maj_weight = cur_weight
                # reset runnig values for current particle
                cur_particle_id = hit.particle_id
                cur_nhits = 1
                cur_weight = hit.weight
            # hit belongs to the same particle within the same reconstructed track
            else:
                cur_nhits += 1
                cur_weight += hit.weight

        # last track is not handled inside the loop
        if maj_nhits < cur_nhits:
            maj_particle_id = cur_particle_id
            maj_nhits = cur_nhits
            maj_weight = cur_weight
        # store values for the last track
        tracks.append((rec_track_id, rec_nhits, maj_particle_id,
            particles_nhits[maj_particle_id], maj_nhits, maj_weight / total_weight))

        cols = ['track_id', 'nhits',
                'major_particle_id', 'major_particle_nhits',
                'major_nhits', 'major_weight']
        return pd.DataFrame.from_records(tracks, columns=cols)
    
    def _analyze_tracks2(self, truth, submission):
        
        merged_df = pd.merge(submission, truth, on='hit_id', how='inner')
        #merged_df.query('particle_id != 0', inplace=True)

        total_weight = truth['weight'].sum()
        grouped = merged_df.groupby(['track_id', 'particle_id']).agg(
            hits_in_track=('hit_id', 'count'),
            major_particle_weight=('weight', 'first')  # Assuming weight is the same for the same particle_id
        ).reset_index()

        most_present_particle = grouped.loc[grouped.groupby('track_id')['hits_in_track'].idxmax()]
        track_hits = merged_df.groupby('track_id').size().reset_index(name='number_of_hits')
        result = pd.merge(most_present_particle, track_hits, on='track_id', how='inner')
        result['major_weight'] = (result['major_particle_weight'] * result['hits_in_track']) / total_weight

        result = result.rename(columns={
            'hits_in_track': 'major_particle_nhits',
            'particle_id': 'major_particle_id'
        })

        # Select and reorder columns
        result = result[['track_id', 'number_of_hits', 'major_particle_id',
                         'major_particle_nhits', 'major_particle_weight', 'major_weight']]

        return result

    def evaluate_tracking(self, event: int = 0, input_dataframe_file: str = None):
        '''
            Evaluate the tracking for a single event. The evaluation is done following the procedure
            described in the competition documentation. 
        '''

        print(tc.GREEN+tc.BOLD+'Evaluating tracking'+tc.RESET)

        PID_IDX = self.data_handler.PID_IDX

        dataframe = None

        if input_dataframe_file is not None:
            dataframe = pd.read_csv(input_dataframe_file)
        else:
            # create a dataframe with hit_id, particle_id, weight and a dataframe with hit_id, track_id
            dataframe = pd.DataFrame(self.data_handler.dataset, columns=[
                                     'x', 'y', 'z', 'cl_size', 'amplitude', 'particle_id', 'weight', 'event', 'hit_id', 'unique_module_id'])
            dataframe['particle_id'] = self.data_handler.dataset[:, PID_IDX]
            dataframe['track_id'] = self.track_ids

        self.track_dataframe = dataframe

        dataframe.query(f'event == {event}', inplace=True)
        truth = dataframe[['hit_id', 'particle_id', 'weight']].copy()
        submission = dataframe[['hit_id', 'track_id']].copy()

        # Calculate the score

        ######################################################################
        #results = self._analyze_tracks2(truth, submission)
        #score = results['major_weight'].sum()
        #
        #print(f'\t* Score: {score:.4f}')

        # from solution2
        #######################################################################
        #
        #truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
        #df = truth.groupby(['track_id', 'particle_id'])['hit_id'].count().to_frame('count_both').reset_index()
        #truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])
        #
        #df1 = df.groupby(['particle_id'])['count_both'].sum().to_frame('count_particle').reset_index()
        #truth = truth.merge(df1, how='left', on='particle_id')
        #df1 = df.groupby(['track_id'])['count_both'].sum().to_frame('count_track').reset_index()
        #truth = truth.merge(df1, how='left', on='track_id')
        #truth['count_both'] *= 2
        #score = truth[(truth['count_both'] > truth['count_particle']) & (truth['count_both'] > truth['count_track'])]['weight'].sum()
        #particles = truth[(truth['count_both'] > truth['count_particle']) & (truth['count_both'] > truth['count_track'])]['particle_id'].unique()
        #
        #print(f'\t* Score: {score:.4f}')

        #########################################################################
        print('df after event sel\n', dataframe.describe(), '')
        print('truth\n', truth.describe(), '')
        print('submission\n', submission.describe(), '')
        
        analized_tracks = self._analyze_tracks(truth, submission)
        purity_rec = np.divide(analized_tracks['major_nhits'], analized_tracks['nhits'])
        purity_maj = np.divide(analized_tracks['major_nhits'], analized_tracks['major_particle_nhits'])
        good_track_mask = (purity_rec > 0.5) & (purity_maj > 0.5)
        #score = analized_tracks[good_track_mask]['major_weight'].sum()
        score = analized_tracks['major_weight'].sum()
        
        print(f'\t* Score: {score:.4f}')
        print(f'\t* Purity rec: {np.mean(purity_rec[good_track_mask]):.4f}')
        print(f'\t* Purity maj: {np.mean(purity_maj[good_track_mask]):.4f}')
        
    def save_tracks(self, output_file: str):
        '''
            Save the tracks to a file
        '''

        print('Saving tracks to '+tc.CYAN+tc.UNDERLINE+f'{output_file}'+tc.RESET)
        self.track_dataframe.to_csv(output_file, index=False)