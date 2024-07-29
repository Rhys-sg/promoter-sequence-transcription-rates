import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    return [mapping[nucleotide] for nucleotide in sequence]

def encode_sequences(X, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in X)
    encoded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(encoded_sequences), max_length

def make_predictions(model, X):
    predictions_array = model.predict(np.array(X))[:, 0]
    predictions = pd.DataFrame(predictions_array, columns=['Value'])
    
    return predictions

def plot_scatter(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.scatter(observed, predicted, color='blue', alpha=0.5, label='Data points')

    plt.title('Observed vs. Our Prediction')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    # input_shape = (87, 4)
    model = load_model('v2/Models/CNN_2_1.keras')

    X_test = pd.read_csv('v2/Data/Brewster.csv')['Sequence']
    X_test = X_test.apply(lambda x: 'TTTTCTATCTACGTAC' + x + 'CTCTACCTTAGTTTGTACGTT') # Add UP, ITR sequences
    X_test, max_length = encode_sequences(X_test, 87)

    # y_pred = pd.read_csv('v2/data/Brewster.csv')['Energy (kT)']
    y_pred = pd.read_csv('v2/data/Brewster.csv')['Energy (AU)']

    predictions = make_predictions(model, X_test)

    plot_scatter(y_pred, predictions['Value'])
