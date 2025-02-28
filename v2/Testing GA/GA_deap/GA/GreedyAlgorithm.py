import random
import numpy as np
import tensorflow as tf
import os
from .CNN import CNN

class GreedyAlgorithm:
    '''
    Greedy search algorithm to optimize sequences.
    Finds the optimal single nucleotide mutation, then iterates until it reaches a local optimal.

    '''

    def __init__(self, cnn_model_path, masked_sequence, target_expression, max_iter=100, seed=None):
        if seed is not None:
            self._set_seed(seed)

        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.max_iter = max_iter
        self.nucleotides = [
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1]   # T
        ]

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]

    def _mutate_sequence(self, sequence, index, new_nucleotide):
        mutated_sequence = np.array(sequence, copy=True)
        mutated_sequence[index] = new_nucleotide
        return mutated_sequence

    def _evaluate_sequence(self, sequence):
        prediction = self.cnn.predict([sequence], use_cache=False)[0]
        error = abs(self.target_expression - prediction)
        return prediction, error

    def run(self):
        '''Run the greedy search algorithm iteratively.'''
        current_sequence = np.array(self.masked_sequence, copy=True)
        current_prediction, current_error = self._evaluate_sequence(current_sequence)

        best_sequence = self.cnn.reverse_one_hot_sequence(current_sequence)
        best_prediction = current_prediction
        best_error = current_error

        for _ in range(self.max_iter):
            improved = False

            for idx in self.mask_indices:
                for nucleotide in self.nucleotides:
                    if np.allclose(current_sequence[idx], nucleotide):
                        continue  # Skip if it's the same nucleotide

                    mutated_sequence = self._mutate_sequence(current_sequence, idx, nucleotide)
                    mutated_prediction, mutated_error = self._evaluate_sequence(mutated_sequence)

                    if mutated_error < best_error:
                        best_sequence = self.cnn.reverse_one_hot_sequence(mutated_sequence)
                        best_prediction = mutated_prediction
                        best_error = mutated_error
                        current_sequence = mutated_sequence
                        improved = True  # Mark that we found a better solution

            if not improved:
                print("No improvement found. Stopping early.")
                break

            if best_error == 0:
                print("Perfect match found. Stopping early.")
                break

        return best_sequence, best_prediction, best_error
