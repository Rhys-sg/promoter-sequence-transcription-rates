import random
import numpy as np
import tensorflow as tf
import os
from collections import deque
from .CNN import CNN

class TabuSearch:
    '''
    Tabu Search algorithm to optimize sequences.
    '''
    def __init__(self, cnn_model_path, masked_sequence, target_expression,
                 max_iter=1000, tabu_size=50, neighborhood_size=20, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        self.neighborhood_size = neighborhood_size

        self.tabu_list = deque(maxlen=self.tabu_size)

        self.nucleotides = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ])

        self.prediction_history = []
        self.error_history = []
        self.infill_history = []

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _initialize_random_sequence(self):
        sequence = np.array(self.masked_sequence, copy=True)
        for idx in self.mask_indices:
            sequence[idx] = random.choice(self.nucleotides)
        return sequence

    def _generate_neighbors(self, sequence):
        neighbors = []
        for _ in range(self.neighborhood_size):
            neighbor = np.array(sequence, copy=True)
            idx = random.choice(self.mask_indices)
            new_nuc = random.choice([n for n in self.nucleotides if not np.array_equal(n, neighbor[idx])])
            neighbor[idx] = new_nuc
            neighbors.append(neighbor)
        return neighbors

    def _evaluate_sequence(self, sequence):
        prediction = self.cnn.predict([sequence], use_cache=False)[0]
        error = abs(self.target_expression - prediction)
        return prediction, error

    def run(self):
        '''Run the Tabu Search algorithm.'''
        current_sequence = self._initialize_random_sequence()
        current_prediction, current_error = self._evaluate_sequence(current_sequence)

        best_sequence = current_sequence
        best_prediction = current_prediction
        best_error = current_error

        for iteration in range(self.max_iter):
            neighbors = self._generate_neighbors(current_sequence)
            best_neighbor = None
            best_neighbor_error = float('inf')
            best_neighbor_prediction = None

            for neighbor in neighbors:
                seq_key = tuple(map(tuple, neighbor))
                if seq_key in self.tabu_list:
                    continue
                prediction, error = self._evaluate_sequence(neighbor)
                if error < best_neighbor_error:
                    best_neighbor = neighbor
                    best_neighbor_prediction = prediction
                    best_neighbor_error = error

            if best_neighbor is None:
                break  # All neighbors were in tabu list

            current_sequence = best_neighbor
            current_prediction = best_neighbor_prediction
            current_error = best_neighbor_error

            self.tabu_list.append(tuple(map(tuple, current_sequence)))

            if current_error < best_error:
                best_sequence = current_sequence
                best_prediction = current_prediction
                best_error = current_error

            self.prediction_history.append(best_prediction)
            self.error_history.append(best_error)
            self.infill_history.append(best_sequence)

            if best_error == 0:
                break

        return self.cnn.reverse_one_hot_sequence(best_sequence), best_prediction, best_error
