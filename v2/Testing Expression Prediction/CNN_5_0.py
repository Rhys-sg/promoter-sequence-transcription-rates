import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    return df

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    encoding = [mapping[nucleotide.upper()] for nucleotide in sequence]
    return encoding

def preprocess_sequences(X):
    max_length = max(len(seq) for seq in X)
    padded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(padded_sequences), max_length

def reshape_model_input(X):
    return np.array([[x, x, x, x] for x in X.values]).reshape(-1, 1, 4)

def concatenate_inputs(array1, array2):
    return np.concatenate((array1, array2), axis=1)

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
    return history

def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    return loss

def save_model(model, filename):
    if not filename.endswith('.keras'):
        filename += '.keras'
    model.save('../Models/' + filename)

def load_and_predict(filename, X):
    if not filename.endswith('.keras'):
        filename += '.keras'
    model = load_model('../Models/' + filename)
    predictions_array = model.predict(np.array(X))[:, 0]
    return pd.DataFrame(predictions_array, columns=['Value'])

def ravel(array):
    return np.ravel(array)

def plot_kde(df, predicted):
    sns.kdeplot(df['Normalized Observed log(TX/Txref)'], fill=True, color='blue', label='Observed log(TX/Txref)')
    sns.kdeplot(predicted, fill=True, color='green', label='Our Prediction')
    plt.title('Kernel Density Plot')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def plot_scatter(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.scatter(observed, predicted, color='blue', alpha=0.5, label='Data points')
    min_val = min(min(observed), min(predicted))
    max_val = max(max(observed), max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x line')
    plt.title('Observed log(TX/Txref) vs. Our Prediction')
    plt.xlabel('Observed log(TX/Txref)')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hexbin(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.hexbin(observed, predicted, gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(label='Counts')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y = x line')
    plt.title('Observed log(TX/Txref) vs. Our Prediction')
    plt.xlabel('Observed log(TX/Txref)')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_path = '../Data/41467_2022_32829_MOESM5_ESM.csv'
    df = load_and_preprocess_data(file_path)
    X, y = combine_columns(df)
    X, max_length = preprocess_sequences(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_cnn_model(X.shape[1:])
    
    history = train_model(model, X_train, y_train, X_test, y_test)
    loss = evaluate_model(model, X_test, y_test)
    save_model(model, 'CNN_1_3_1.keras')
    
    predicted = load_and_predict('CNN_1_3_1.keras', X)
    observed = df['Normalized Observed log(TX/Txref)'].values
    
    plot_kde(df, predicted)
    plot_scatter(observed, np.ravel(predicted))
    plot_hexbin(observed, np.ravel(predicted))

if __name__ == "__main__":
    main()