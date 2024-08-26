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
    plt.title('Energy (AU) vs. Our Predicted dG')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter_labeled(observed, predicted, names):
    plt.figure(figsize=(10, 6))

    unique_names = names.unique()
    colors = plt.cm.get_cmap('tab20', len(unique_names))
    color_map = {name: colors(i) for i, name in enumerate(unique_names)}
    
    for name in unique_names:
        idx = names == name
        plt.scatter(observed[idx], predicted[idx], color=color_map[name], alpha=0.7, label=name)
        for i in np.where(idx)[0]:
            plt.text(observed[i], predicted[i], names[i], fontsize=8, ha='right')
    
    # plt.plot([max(observed), min(observed)], [max(predicted), min(predicted)], color='black', linestyle='--')
    plt.title('Energy (AU) vs. Our Predicted dG')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.grid(True)
    plt.show()

def plot_scatter_grouped(observed, predicted, names):
    plt.figure(figsize=(10, 6))

    variant_groups = {
        'UV5': 'UV',
        'WT': 'WT',
        'WTDL10': 'WT-DL',
        'WTDL20': 'WT-DL',
        'WTDL20v2': 'WT-DL',
        'WTDL30': 'WT-DL',
        'WTDR30': 'WT-DR',
        '5DL1': '5DL',
        '5DL5': '5DL',
        '5DL10': '5DL',
        '5DL20': '5DL',
        '5DL30': '5DL',
        '5DR1': '5DR',
        '5DR1v2': '5DR',
        '5DR5': '5DR',
        '5DR10': '5DR',
        '5DR20': '5DR',
        '5DR30': '5DR'
    }
    
    unique_variants = list(set(variant_groups.values()))
    colors = plt.cm.get_cmap('tab20', len(unique_variants))
    color_map = {variant: colors(i) for i, variant in enumerate(unique_variants)}
    
    for variant in unique_variants:
        idx = names.map(variant_groups) == variant
        plt.scatter(observed[idx], predicted[idx], color=color_map[variant], alpha=0.7, label=variant)
        for i in np.where(idx)[0]:
            plt.text(observed[i], predicted[i], names[i], fontsize=8, ha='right')

    # plt.plot([max(observed), min(observed)], [max(predicted), min(predicted)], color='black', linestyle='--')
    plt.title('Energy (AU) vs. Our Predicted dG')
    plt.xlabel('Observed')
    plt.ylabel('Our Prediction')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    model = load_model('v2/Models/CNN_2_1.keras')

    X_test = pd.read_csv('v2/Data/Brewster.csv')['Sequence']
    X_test = X_test.apply(lambda x: 'TTTTCTATCT' + x + 'CTCTACCTTAGTTTGTACGTT')
    X_test, max_length = encode_sequences(X_test, 87)

    y_pred = pd.read_csv('v2/Data/Brewster.csv')['Energy (AU)']
    names = pd.read_csv('v2/Data/Brewster.csv')['Name']

    predictions = make_predictions(model, X_test)

    plot_scatter(y_pred, predictions['Value'])
    plot_scatter_labeled(y_pred, predictions['Value'], names)
    plot_scatter_grouped(y_pred, predictions['Value'], names)
