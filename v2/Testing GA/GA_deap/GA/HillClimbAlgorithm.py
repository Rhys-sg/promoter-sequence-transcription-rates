import random
import numpy as np
import tensorflow as tf
import os
from .CNN import CNN

class HillClimbAlgorithm:
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

    def _mutate_sequences(self, sequence):
        ''' Generate all possible mutations for the masked indices using list comprehension '''
        original_sequence = np.array(sequence, copy=True)
        
        return np.array([
            (mutated_sequence := original_sequence.copy(), mutated_sequence.__setitem__(idx, nucleotide))[0]
            for idx in self.mask_indices
            for nucleotide in self.nucleotides
            if not np.allclose(original_sequence[idx], nucleotide)
        ])

    def _evaluate_sequences(self, sequences):
        ''' Predict all mutated sequences at once '''
        predictions = self.cnn.predict(sequences, use_cache=False)
        errors = np.abs(self.target_expression - predictions)
        return predictions, errors

    def run(self):
        '''Run the greedy search algorithm iteratively while tracking history.'''
        current_sequence = np.array(self.masked_sequence, copy=True)
        current_prediction, current_error = self._evaluate_sequences([current_sequence])

        best_sequence = self.cnn.reverse_one_hot_sequence(current_sequence)
        best_prediction = current_prediction[0]
        best_error = current_error[0]

        # Tracking history
        self.prediction_history = [best_prediction]
        self.error_history = [best_error]
        self.sequence_history = [best_sequence]

        for _ in range(self.max_iter):
            mutated_sequences = self._mutate_sequences(current_sequence)

            if len(mutated_sequences) == 0:
                break

            predictions, errors = self._evaluate_sequences(mutated_sequences)

            # Find the best mutation
            min_error_idx = np.argmin(errors)

            if errors[min_error_idx] < best_error:
                best_sequence = self.cnn.reverse_one_hot_sequence(mutated_sequences[min_error_idx])
                best_prediction = predictions[min_error_idx]
                best_error = errors[min_error_idx]
                current_sequence = mutated_sequences[min_error_idx]  # Update the current sequence
            else:
                break

            # Store history
            self.prediction_history.append(best_prediction)
            self.error_history.append(best_error)
            self.sequence_history.append(best_sequence)

            if best_error == 0:
                break

        return best_sequence, best_prediction, best_error
