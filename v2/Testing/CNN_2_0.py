import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

class SuppressOutput:
    def __enter__(self):
        self.stdout_original = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout_original


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR']].astype(str).agg(''.join, axis=1)
    y = df['Observed']
    return X, y, df

def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    return [mapping[nucleotide] for nucleotide in sequence]

def encode_sequences(X):
    max_length = max(len(seq) for seq in X)
    encoded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(encoded_sequences), max_length

def build_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=4, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
    return history

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss

def save_model(model, filename):
    model.save(filename)

def load_saved_model(filename):
    model = load_model(filename)
    return model

def make_predictions(model, X):
    predictions_array = model.predict(np.array(X))[:, 0]
    predictions = pd.DataFrame(predictions_array, columns=['Value'])
    
    return predictions

def plot_kde(observed, predicted):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(observed, fill=True, color='blue', label='Observed')
    sns.kdeplot(predicted, fill=True, color='green', label='Our Prediction')
    plt.title('Kernel Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_scatter(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.scatter(observed, predicted, color='blue', alpha=0.5, label='Data points')

    # Add y=x line
    min_val = min(min(observed), min(predicted))
    max_val = max(max(observed), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x line')

    plt.title('Observed vs. Our Prediction')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_scatter_density(observed, predicted):
    plt.figure(figsize=(10, 6))

    # Hexbin plot
    plt.hexbin(observed, predicted, gridsize=50, cmap='Blues', mincnt=1)
    
    # Add color bar
    plt.colorbar(label='Counts')

    # Add y=x line
    min_val = min(min(observed), min(predicted))
    max_val = max(max(observed), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x line')

    plt.title('Observed vs. Our Prediction')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()
