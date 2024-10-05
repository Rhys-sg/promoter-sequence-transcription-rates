import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    return df

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

# THIS DOES NOT CONSIDER VARIATION IN THE REMOVED SECTION LENGTH
def preprocess_X_y(df, num_augmentations=1):
    sequences, expressions = combine_columns(df)

    X_sequences = []
    X_expressions = []
    X_len_removed = []
    y_missing = []

    for sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(1, 10)
            masked_sequence, missing_element = remove_section(sequence, len_removed)
            X_sequences.append(masked_sequence)
            X_expressions.append(expression)
            X_len_removed.append(len_removed)
            y_missing.append(missing_element)

    X_sequences = padded_one_hot_encode_all(X_sequences)
    y_missing = padded_one_hot_encode_all(y_missing)

    return X_sequences, X_expressions, X_len_removed, y_missing

def remove_section(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + '_' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element

def padded_one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'C': [0, 0, 1, 0, 0], 'G': [0, 0, 0, 1, 0], 
               '_': [0, 0, 0, 0, 1],  # Placeholder for missing section
               '0': [0, 0, 0, 0, 0]}  # Placeholder for padding
    return [mapping[nucleotide.upper()] for nucleotide in sequence]

def padded_one_hot_encode_all(sequences, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    padded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in sequences]
    return np.array(padded_sequences)