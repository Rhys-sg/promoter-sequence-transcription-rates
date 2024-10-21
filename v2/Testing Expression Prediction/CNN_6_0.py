import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model  # type: ignore
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Activation  # type: ignore
from keras.optimizers import Adam  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from keras_tuner import HyperModel, RandomSearch  # type: ignore

def load_features(file_path):
    df = pd.read_csv(file_path)
    y = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']].abs())
    X = preprocess_sequences(df[['Promoter Sequence']].astype(str).agg(''.join, axis=1))
    return X, y

def preprocess_sequences(X, max_length=150):
    return np.array([padded_one_hot_encode(seq.zfill(max_length)) for seq in X])

def padded_one_hot_encode(sequence):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], '0': [0, 0, 0, 0]}
    return np.array([mapping[nucleotide.upper()] for nucleotide in sequence])

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, loss):
        self.input_shape = input_shape
        self.loss = loss

    def build(self, hp):
        model = Sequential()
        
        # Tune number of layers
        for i in range(hp.Int('num_layers', 1, 3)):
            # Tune filters, kernel size, activation, and pooling
            model.add(Conv1D(
                filters=hp.Choice(f'filters_{i}', [32, 64, 128]),
                kernel_size=hp.Choice(f'kernel_size_{i}', [3, 5, 7]),
                strides=hp.Choice(f'strides_{i}', [1, 2]),
                padding=hp.Choice(f'padding_{i}', ['same', 'valid']),
                activation=hp.Choice(f'activation_{i}', ['relu', 'tanh'])
            ))
            model.add(MaxPooling1D(pool_size=hp.Choice(f'pool_size_{i}', [2, 3])))

        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', 32, 128, step=32), activation='relu'))
        model.add(Dense(1, activation='linear'))

        # Compile model with tuned learning rate
        model.compile(
            optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss=self.loss
        )
        return model

def train_best_model(X_train, y_train, X_test, y_test, input_shape, loss, max_trials=5, epochs=2, batch_size=32):
    tuner = RandomSearch(
        CNNHyperModel(input_shape=input_shape, loss=loss),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='cnn_tuning',
        project_name='cnn_hyperparam_search'
    )

    tuner.search_space_summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop])

    tuner.results_summary()
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model

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

if __name__ == "__main__":

    # Hyperparameter tuning variables
    max_trials = 10
    
    # Training Hyperparameters
    epochs = 2
    batch_size = 32
    loss = 'mean_squared_error'

    X_train, y_train = load_features('v2/Data/Train Test/train_data.csv')
    X_test, y_test = load_features('v2/Data/Train Test/test_data.csv')

    # Hyperparameter tuning
    best_model = train_best_model(X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  X_train.shape[1:],
                                  loss,
                                  max_trials,
                                  epochs=epochs,
                                  batch_size=batch_size)

    # Save the best model
    model_path = 'v2/Models/CNN_6_0.keras'
    best_model.save(model_path)

    # Load, predict, and evaluate the best model
    y_pred = load_and_predict(model_path, X_test)
    mse, rmse, mae, r2 = calc_metrics(y_test, y_pred)

    print('MSE: ', mse)
    print('RMSE: ', rmse)
    print('MAE: ', mae)
    print('R2: ', r2)