import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization , Activation # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from keras_tuner import HyperModel, BayesianOptimization
import seaborn as sns
import matplotlib.pyplot as plt

def load_features(file_path):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']].abs())
    X = preprocess_sequences(df[['Promoter Sequence']].astype(str).agg(''.join, axis=1))
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_sequences(X, max_length=150):
    return np.array([padded_one_hot_encode(seq.zfill(max_length)) for seq in X])

def padded_one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '0': [0, 0, 0, 0]}
    return np.array([mapping[nucleotide.upper()] for nucleotide in sequence])

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, loss, hyperparam_ranges):
        self.input_shape = input_shape
        self.loss = loss
        self.hyperparam_ranges = hyperparam_ranges

    def build(self, hp):
        model = Sequential()
        
        # Number of convolutional layers
        for i in range(hp.Int('num_layers', 
                              self.hyperparam_ranges['num_layers'][0], 
                              self.hyperparam_ranges['num_layers'][1])):
            model.add(Conv1D(
                filters=hp.Choice(f'filters_{i}', self.hyperparam_ranges['filters']),
                kernel_size=hp.Choice(f'kernel_size_{i}', self.hyperparam_ranges['kernel_size']),
                strides=hp.Choice(f'strides_{i}', self.hyperparam_ranges['strides']),
                activation=None
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=hp.Choice(f'pool_size_{i}', self.hyperparam_ranges['pool_size'])))
            model.add(Activation(hp.Choice(f'activation_{i}', self.hyperparam_ranges['activation'])))

        model.add(Flatten())
        
        # Dense layer with configurable units
        model.add(Dense(units=hp.Int('dense_units', 
                                     self.hyperparam_ranges['dense_units'][0], 
                                     self.hyperparam_ranges['dense_units'][1], 
                                     step=self.hyperparam_ranges['dense_units'][2]), 
                        activation='relu'))
        
        # Final output layer
        model.add(Dense(1, activation='linear'))

        # Compile the model with the configurable learning rate
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 
                                    self.hyperparam_ranges['learning_rate'][0], 
                                    self.hyperparam_ranges['learning_rate'][1], 
                                    sampling='log')),
            loss=self.loss
        )
        return model

def train_best_model(name, search_dir, X_train, y_train, X_val, y_val, input_shape, loss, max_trials, epochs, batch_size, hyperparam_ranges):
    tuner = BayesianOptimization(
        CNNHyperModel(input_shape=input_shape, loss=loss, hyperparam_ranges=hyperparam_ranges),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory=search_dir,
        project_name=f'{name}_bayesian_search'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                 validation_data=(X_val, y_val), callbacks=[early_stop])

    return tuner.get_best_models(num_models=1)[0], tuner.get_best_hyperparameters(num_trials=1)[0]

def calc_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def load_and_predict(model_path, X):
    model = load_model(model_path)
    predictions_array = model.predict(np.array(X))[:, 0]
    return pd.DataFrame(predictions_array, columns=['Value'])

def plot_kde(df, predicted):
    sns.kdeplot(df['Normalized Expression'], fill=True, color='blue', label='Normalized Expression')
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
    plt.xlabel('Observed Expression')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hexbin(observed, predicted):
    plt.figure(figsize=(10, 6))
    plt.hexbin(observed, predicted, gridsize=50, cmap='Blues', mincnt=1)
    plt.colorbar(label='Counts')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y = x line')
    plt.xlabel('Observed Expression')
    plt.ylabel('Our Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()