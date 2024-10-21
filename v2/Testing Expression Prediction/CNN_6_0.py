import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense # type: ignore
from keras.optimizers import Adam # type: ignore
import seaborn as sns
import matplotlib.pyplot as plt

def load_features(file_path):
    df = pd.read_csv(file_path)
    y = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']])
    X = preprocess_sequences(df[['Promoter Sequence']].astype(str).agg(''.join, axis=1))
    return X, y

def preprocess_sequences(X, max_length=150):
    padded_sequences = [padded_one_hot_encode('0' * (max_length - len(seq)) + seq) for seq in X]
    return np.array(padded_sequences)

def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], '0': [0,0,0,0]}
    encoding = [mapping[nucleotide.upper()] for nucleotide in sequence]
    return encoding

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

def train_model(model, X_train, y_train, X_test, y_test, epochs=150, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

def load_and_predict(model_path, X):
    model = load_model(model_path)
    predictions_array = model.predict(np.array(X))[:, 0]
    return pd.DataFrame(predictions_array, columns=['Value'])

def calc_metrics(y_test, y_pred):
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - mse / np.var(y_test)
    return mse, rmse, mae, r2

if __name__ == "__main__":
    model_path = '../Models/CNN_6_0.keras'
    X_train, y_train = load_features('../Data/Train Test/train_data.csv')
    X_test, y_test = load_features('../Data/Train Test/test_data.csv')
    model = build_cnn_model(X_train.shape[1:])
    history = train_model(model, X_train, y_train, X_test, y_test)

    model.save(model_path)
    y_pred = load_and_predict(model_path, X_test)

    mse, rmse, mae, r2 = calc_metrics(y_test, y_pred)

    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAE: ', mae)
    print('R2: ', r2)