import random
import math
import numpy as np
import tensorflow as tf
import os
from .CNN import CNN

class SimulatedAnnealing:
    '''
    Simulated Annealing search algorithm to optimize sequences.
    '''
    def __init__(self, cnn_model_path, masked_sequence, target_expression, 
                 initial_temperature=10.0, cooling_rate=0.99, max_iter=1000, seed=None):
        if seed is not None:
            self._set_seed(seed)
        
        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
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
    
    def _initialize_random_sequence(self):
        sequence = np.array(self.masked_sequence, copy=True)
        for idx in self.mask_indices:
            sequence[idx] = random.choice(self.nucleotides)
        return sequence
    
    def _mutate_sequence(self, sequence):
        mutated_sequence = np.array(sequence, copy=True)
        idx = random.choice(self.mask_indices)
        mutated_sequence[idx] = random.choice(self.nucleotides)
        return mutated_sequence
    
    def _evaluate_sequence(self, sequence):
        prediction = self.cnn.predict([sequence], use_cache=False)[0]
        error = abs(self.target_expression - prediction)
        return prediction, error
    
    def run(self):
        '''Run the simulated annealing algorithm.'''
        current_sequence = self._initialize_random_sequence()
        current_prediction, current_error = self._evaluate_sequence(current_sequence)
        
        best_sequence = current_sequence
        best_prediction = current_prediction
        best_error = current_error
        
        for iteration in range(self.max_iter):
            new_sequence = self._mutate_sequence(current_sequence)
            new_prediction, new_error = self._evaluate_sequence(new_sequence)
            
            error_difference = new_error - current_error
            
            if error_difference < 0 or math.exp(-error_difference / self.temperature) > random.random():
                current_sequence = new_sequence
                current_prediction = new_prediction
                current_error = new_error
            
                if current_error < best_error:
                    best_sequence = current_sequence
                    best_prediction = current_prediction
                    best_error = current_error
            
            self.temperature *= self.cooling_rate
            
            if best_error == 0:
                break

            self.prediction_history.append(best_prediction)
            self.error_history.append(best_error)
            self.infill_history.append(best_sequence)

        return self.cnn.reverse_one_hot_sequence(best_sequence), best_prediction, best_error
