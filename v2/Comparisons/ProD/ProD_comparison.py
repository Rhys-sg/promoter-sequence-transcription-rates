import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

def get_predictions(model, X_test):

    predictions = np.array(model.predict(X_test)).flatten()
    
    average_prediction = np.mean(predictions)

    # 0.206 of model variance explained by spacer element
    min_val = average_prediction - 0.103
    max_val = average_prediction + 0.103
    
    def normalize(pred, min_val, max_val):
        return 10 * (pred - min_val) / (max_val - min_val)
    
    normalized_predictions = [normalize(y_pred, min_val, max_val) for y_pred in predictions]
    
    return [round(pred) for pred in normalized_predictions]

def get_preprocess_X_test(df):
    ProD_upstream = 'TTGCTGGATAACTTTACG'
    ProD_downstream = 'TATAATATTCAGG'
    UP_extension = 'TTT'
    TR_extension = 'CTCTACCTTAGTTTGTACGTT'

    X_test = [UP_extension + ProD_upstream + spacer[0] + ProD_downstream + TR_extension for spacer in df[['spacer']].values]
    max_length = 87
    X_test_encoded, max_length = encode_sequences(X_test, max_length)
    
    print(f"X_test shape: {X_test_encoded.shape}")
    print(f"X_test data type: {X_test_encoded.dtype}")
    
    return X_test_encoded

# Ensure the data type is float32
def encode_sequences(X, max_length=None):
    if max_length is None:
        max_length = max(len(seq) for seq in X)
    encoded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(encoded_sequences, dtype=np.float32), max_length


def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    return [mapping[nucleotide] for nucleotide in sequence]


def plot_confusion_matrices(y_true, y_pred1, y_pred2, model_name1, model_name2):
    cm1 = confusion_matrix(y_true, y_pred1, labels=np.arange(0, 11))
    cm2 = confusion_matrix(y_true, y_pred2, labels=np.arange(0, 11))
    cm1_normalized = cm1.astype(float) / cm1.sum(axis=1)[:, np.newaxis] * 100
    cm2_normalized = cm2.astype(float) / cm2.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(cm1_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, 
                xticklabels=np.arange(0, 11), yticklabels=np.arange(0, 11), ax=axes[0])
    axes[0].set_xlabel('Predicted Labels')
    axes[0].set_ylabel('True Labels')
    axes[0].set_title(f'Confusion Matrix for {model_name1}')
    
    sns.heatmap(cm2_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True, 
                xticklabels=np.arange(0, 11), yticklabels=np.arange(0, 11), ax=axes[1])
    axes[1].set_xlabel('Predicted Labels')
    axes[1].set_ylabel('True Labels')
    axes[1].set_title(f'Confusion Matrix for {model_name2}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('v2/Data/ProD_sigma70_spacer_data.csv')

    X_test = get_preprocess_X_test(df)

    scaler = MinMaxScaler()
    y_test = scaler.fit_transform(df[['assumed_observed']].astype(float)) * 10

    model_synth_cure = load_model('v2/Models/CNN_2_1.keras')
    df['Synth_CURE_predicted'] = get_predictions(model_synth_cure, X_test)

    assumed_observed = np.ravel(df['assumed_observed'])
    synth_pred = np.ravel(df['Synth_CURE_predicted'])
    prod_pred = np.ravel(df['ProD_predicted'])

    plot_confusion_matrices(assumed_observed, synth_pred, prod_pred, 'Synth_CURE', 'ProD')
