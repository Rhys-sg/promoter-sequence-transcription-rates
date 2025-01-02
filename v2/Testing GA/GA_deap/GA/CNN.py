import numpy as np
import torch
from keras.models import load_model  # type: ignore

class CNN:
    '''
    This class is a wrapper for a convolutional neural network (CNN) model that predicts the transcription rate of a given sequence.
    It includes methods for preprocessing sequences, predicting the transcription rate of sequences, and one-hot/reverse one-hot encoding sequences.

    The main method is predict, which takes a list of sequences and returns a np.array() of predictions.
    Sequences can be as one-hot encoded as tuples (used for caching).
    Use preprocess, one_hot_sequence, and reverse_one_hot_sequence for encoding/decoding sequences.

    '''
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.input_length = self.model.input_shape[1]
        self.cache = {}
    
    def predict(self, sequences, use_cache=True):
        if use_cache:
            return self._cached_predict(sequences)
        return self._predict(sequences)
    
    def one_hot_sequence(self, sequence):
        mapping = {
            'A': (1, 0, 0, 0),
            'C': (0, 1, 0, 0),
            'G': (0, 0, 1, 0),
            'T': (0, 0, 0, 1),
            '0': (0, 0, 0, 0),
            'N': (0.25, 0.25, 0.25, 0.25)
        }
        return tuple(mapping[nucleotide.upper()] for nucleotide in sequence.zfill(self.input_length))
    
    def reverse_one_hot_sequence(self, one_hot_sequence, pad=False):
        mapping = {
            (1, 0, 0, 0): 'A',
            (0, 1, 0, 0): 'C',
            (0, 0, 1, 0): 'G',
            (0, 0, 0, 1): 'T',
            (0, 0, 0, 0): '0' if pad else '',
            (0.25, 0.25, 0.25, 0.25): 'N'
        }
        return ''.join([mapping[tuple(nucleotide)] for nucleotide in one_hot_sequence])
    
    def preprocess(self, sequences):
        return [self.one_hot_sequence(seq) for seq in sequences]

    def _cached_predict(self, sequences):
        predictions = []
        sequences = [self._make_hashable(seq) for seq in sequences]
        to_predict = [seq for seq in sequences if seq not in self.cache]
        if to_predict:
            predictions = self._predict(to_predict)
            for seq, pred in zip(to_predict, predictions):
                self.cache[seq] = pred
        return np.array([self.cache[seq] for seq in sequences])
    
    def _predict(self, sequences):
        tensor_sequences = torch.tensor(
            [list(seq) for seq in sequences],
            dtype=torch.float32
        )
        with torch.no_grad():
            predictions = self.model(tensor_sequences).cpu().numpy().flatten()
        return predictions
    
    @staticmethod
    def _make_hashable(sequence):
        if isinstance(sequence, (list, tuple)):
            return tuple(map(tuple, sequence))
        return sequence
