import random
import math
import numpy as np
import tensorflow as tf
import os
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from .CNN import CNN

class BayesianOptimization:
    '''
    Bayesian optimization algorithm to optimize sequences.
    '''
    
    def __init__(self, cnn_model_path, masked_sequence, target_expression, n_calls=50, seed=None):
        if seed is not None:
            self._set_seed(seed)
        
        self.cnn = CNN(cnn_model_path)
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression
        self.n_calls = n_calls
        
        self.nucleotide_dict = {
            0: [1, 0, 0, 0],  # A
            1: [0, 1, 0, 0],  # C
            2: [0, 0, 1, 0],  # G
            3: [0, 0, 0, 1]   # T
        }
        
        self.space = [Integer(0, 3, name=f"pos_{i}") for i in range(len(self.mask_indices))]
    
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if np.allclose(element, 0.25, atol=1e-9)]
    
    def _reconstruct_sequence(self, infill):
        sequence = np.array(self.masked_sequence, copy=True)
        for idx, char_idx in zip(self.mask_indices, infill):
            sequence[idx] = self.nucleotide_dict[char_idx]
        return sequence
    
    def _objective(self, params):
        """Objective function for Bayesian optimization"""
        infill = params  # Directly using the list of values
        sequence = self._reconstruct_sequence(infill)
        prediction = self.cnn.predict([sequence], use_cache=False)[0]
        error = np.abs(self.target_expression - prediction)
        return error
    
    def run(self):
        '''Run Bayesian optimization to find the best sequence.'''
        result = gp_minimize(self._objective, self.space, n_calls=self.n_calls, random_state=42)
        
        best_infill = result.x
        best_sequence = self._reconstruct_sequence(best_infill)
        best_prediction = self.cnn.predict([best_sequence], use_cache=False)[0]
        best_error = np.abs(self.target_expression - best_prediction)
        
        return self.cnn.reverse_one_hot_sequence(best_sequence), best_prediction, best_error
