import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    return df

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_X_y(df, num_augmentations=1):
    sequences, expressions = combine_columns(df)

    X = []
    y = []

    for sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(1, 10)
            input, output = remove_section_get_features(sequence, len_removed)

            X.append(one_hot_encode(apply_padding(input, 150), expression, len_removed))
            y.append(one_hot_encode(apply_padding(output, 150)))

    return np.array(X), np.array(y)

def remove_section_get_features(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + '_' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element

def apply_padding(sequence, max_length):
    return '0' * (max_length - len(sequence)) + sequence

def one_hot_encode(sequence, expression=None, len_removed=None):
     
    mapping = {'A': [1, 0, 0, 0, 0],
               'T': [0, 1, 0, 0, 0],
               'C': [0, 0, 1, 0, 0],
               'G': [0, 0, 0, 1, 0], 
               '_': [0, 0, 0, 0, 1],  # Placeholder for missing section
               '0': [0, 0, 0, 0, 0]}  # Placeholder for padding
    
    # Only add expression and len_removed if for X data
    if len_removed:
        return [mapping[nucleotide.upper()] + [expression] + [len_removed] for nucleotide in sequence]
    else:
        return [mapping[nucleotide.upper()] for nucleotide in sequence]


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(layers.Masking(mask_value=0.0, input_shape=input_shape))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='softmax'))  # Output layer with 5 units for one-hot encoding
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_lstm_model(model, X_train, y_train, X_test, y_test):
    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=10, 
        batch_size=32, 
        validation_data=(X_test, y_test), 
        verbose=1,
        callbacks=[early_stopping]
    )
    
    return history

def evaluate_lstm_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss

def save_model(model, filename):
    model.save('../Models/' + filename)