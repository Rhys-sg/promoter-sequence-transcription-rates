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

    # plt.plot([max(observed), min(observed)], [max(predicted), min(predicted)], color='red', linestyle='--')
    plt.title('Observed Expression level vs. Our Predicted dG')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter_density(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.hexbin(observed, predicted, gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(label='Counts')

    plt.title('Observed vs. Our Prediction')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('v2/Data/rlp5Min_SplitVariants_Processed.csv')
    model = load_model('v2/Models/CNN_4_0.keras')

    for i, row in df.iterrows():
        if len(row['sub_variant']) < 30:
            df.drop(i, inplace=True)

    X_test = df['sub_variant']
    y_test = df['log_transformed']

    X_test, max_length = encode_sequences(X_test, model.input_shape[1])

    y_pred = make_predictions(model, X_test)

    plot_scatter(y_test, y_pred['Value'])
