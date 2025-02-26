import random
import math
import numpy as np
import tensorflow as tf
import os
from .CNN import CNN

class RandomSearchAlgorithm:
    '''
    Random search algorithm to optimize sequences.
    '''

    def __init__(self, cnn_model_path, masked_sequence, target_expression, batch_size=16384, max_iter=100000, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_iter = max_iter
        self.batch_size = batch_size

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _random_sequence(self):
        nucleotides = np.array([
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ])
        return [random.choice(nucleotides) for _ in range(len(self.mask_indices))]

    def _reconstruct_sequence(self, infill):
        sequence = np.array(self.masked_sequence, copy=True)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return tuple(sequence)
    
    def _run_batch(self, batch):
        predictions = self.cnn.predict(batch, use_cache=False)
        errors = np.abs(self.target_expression - predictions)
        best_idx = np.argmin(errors)
        best_sequence = self.cnn.reverse_one_hot_sequence(batch[best_idx])
        best_prediction = predictions[best_idx]
        best_error = errors[best_idx]
        return best_sequence, best_prediction, best_error

    def run(self):
        '''Run the random search algorithm with max_iter support.'''
        best_sequence = None
        best_prediction = None
        best_error = float('inf')
        evaluated_sequences = 0

        while evaluated_sequences < self.max_iter:
            batch_size = min(self.batch_size, self.max_iter - evaluated_sequences)
            batch = [self._reconstruct_sequence(self._random_sequence()) for _ in range(batch_size)]
            
            current_best_sequence, current_best_prediction, current_best_error = self._run_batch(batch)
            evaluated_sequences += batch_size

            if current_best_error < best_error:
                best_sequence = current_best_sequence
                best_prediction = current_best_prediction
                best_error = current_best_error

            if best_error == 0:
                print("Perfect match found. Stopping early.")
                break

        return best_sequence, best_prediction, best_error
