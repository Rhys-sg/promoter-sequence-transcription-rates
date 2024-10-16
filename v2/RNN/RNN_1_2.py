import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Masking, Concatenate

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

    X_sequence = []
    X_expressions = []
    y = []

    for sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(1, 10)
            input, output = remove_section_get_features(sequence, len_removed)

            X_sequence.append(one_hot_encode(apply_padding(input, 150)))
            X_expressions.append(expression)
            y.append(one_hot_encode(apply_padding(output, 150)))

    return np.array(X_sequence), np.array(X_expressions), np.array(y)

def remove_section_get_features(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + '_' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element

def apply_padding(sequence, max_length):
    return '0' * (max_length - len(sequence)) + sequence

def one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               '_': [0, 0, 0, 0],  # Placeholder for missing section
               '0': [0, 0, 0, 0]}  # Placeholder for padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

# Function to build the LSTM model
def build_model(sequence_length=150, nucleotide_dim=4, expression_dim=1):
    sequence_input = Input(shape=(sequence_length, nucleotide_dim), name='sequence_input')
    expression_input = Input(shape=(sequence_length, expression_dim), name='expression_input')
    combined_input = Concatenate()([sequence_input, expression_input])
    masked_input = Masking(mask_value=0.0)(combined_input)
    lstm_out = LSTM(128, return_sequences=True)(masked_input)
    output = Dense(nucleotide_dim, activation='softmax')(lstm_out)
    model = Model(inputs=[sequence_input, expression_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, X_sequence_train, X_expressions_train, y_train, batch_size=32, epochs=10):
    X_expressions_train = np.expand_dims(X_expressions_train, axis=-1)
    X_expressions_train = np.repeat(X_expressions_train, X_sequence_train.shape[1], axis=1)
    history = model.fit([X_sequence_train, X_expressions_train], y_train, 
                        batch_size=batch_size, epochs=epochs, validation_split=0.2)
    return history

def evaluate_model(model, X_sequence_test, X_expressions_test, y_test):
    X_expressions_test = np.expand_dims(X_expressions_test, axis=-1)
    X_expressions_test = np.repeat(X_expressions_test, X_sequence_test.shape[1], axis=1)
    loss, accuracy = model.evaluate([X_sequence_test, X_expressions_test], y_test)
    
    return loss, accuracy


if __name__ == '__main__':
    
    print('Loading and preprocessing data...')
    file_path = 'v2/Data/combined/LaFleur_supp.csv'
    df = load_and_preprocess_data(file_path)
    
    print('Preparing training and test data...')
    X_sequence, X_expressions, y = preprocess_X_y(df)
    print(X_sequence.shape, X_expressions.shape, y.shape)
    X_sequence_train, X_sequence_test, X_expressions_train, X_expressions_test, y_train, y_test = train_test_split(
        X_sequence, X_expressions, y, test_size=0.2, random_state=42)

    print('Building and training the model...')
    model = build_model(sequence_length=150, nucleotide_dim=4, expression_dim=1)
    
    print('Training the model...')
    train_model(model, X_sequence_train, X_expressions_train, y_train, batch_size=32, epochs=10)
    
    print('Evaluating the model...')
    loss, accuracy = evaluate_model(model, X_sequence_test, X_expressions_test, y_test)