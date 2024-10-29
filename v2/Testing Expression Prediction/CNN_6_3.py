import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
import glob
import json
import os


def load_features(file_path):
    data = pd.read_csv(file_path)
    sequences = data['Promoter Sequence'].values
    targets = data['Normalized Expression'].values
    X = preprocess_sequences(sequences).transpose(0, 2, 1)
    y = targets.astype(np.float32)
    return X, y

def preprocess_sequences(X, max_len=150):
    return np.array([padded_one_hot_encode(seq, max_len) for seq in X])

def padded_one_hot_encode(sequence, max_len):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 
               '0': [0, 0, 0, 0]}  # '0' used for padding
    padded_seq = sequence.ljust(max_len, '0')
    return np.array([mapping.get(nuc, [0, 0, 0, 0]) for nuc in padded_seq])

class CNNModel(nn.Module):
    def __init__(self, input_shape, hp):
        super(CNNModel, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[1]

        for i in range(hp['num_layers']):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hp['filters'][i],
                kernel_size=hp['kernel_size'][i],
                stride=hp['stride'][i]
            )
            self.layers.append(conv)
            self.layers.append(nn.ReLU() if hp['activation'] == 'relu' else nn.Tanh())
            self.layers.append(nn.MaxPool1d(kernel_size=hp['pool_size'][i]))
            in_channels = hp['filters'][i]

        self.layers.append(nn.AdaptiveAvgPool1d(1))
        self.flatten = nn.Flatten()
        conv_out_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(conv_out_size, 1)

    def _get_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape[1:])
        output = dummy_input
        for layer in self.layers:
            output = layer(output)
        return int(np.prod(output.size()))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        return self.fc(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, hp, epochs=100, batch_size=32, learning_rate=0.001, validation_split=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.hp = hp
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.model = CNNModel(input_shape, hp).to(self.device)

    def fit(self, X, y):
        val_size = int(len(X) * self.validation_split)
        X_train, X_val = X[:-val_size], X[-val_size:]
        y_train, y_val = y[:-val_size], y[-val_size:]

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, drop_last=True, shuffle=False, pin_memory=True, num_workers=4)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss().to(self.device)
        early_stopper = EarlyStopper(patience=3, min_delta=10)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}'):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = loss_fn(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation phase
            val_loss = self._validate(val_loader, loss_fn)

            print(f"Epoch {epoch + 1}: Train Loss = {epoch_loss / len(train_loader):.4f}, Val Loss = {val_loss:.4f}\n")

            # Check for early stopping
            if early_stopper.early_stop(val_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

    def _validate(self, val_loader, loss_fn):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = loss_fn(outputs.view(-1), y_batch.view(-1))
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X).squeeze()
        return outputs.cpu().numpy()


def calc_metrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

class PyTorchRegressorCV(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, num_layers=1, filters=(32,), kernel_size=(3,), stride=(1,), 
                 pool_size=(2,), activation='relu', learning_rate=0.001, batch_size=32, epochs=100):
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_size = pool_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.hp = {
            'num_layers': self.num_layers,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'pool_size': self.pool_size,
            'activation': self.activation,
        }

    def fit(self, X, y):
        self.model = PyTorchRegressor(
            input_shape=self.input_shape, 
            hp=self.hp, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            learning_rate=self.learning_rate
        )
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def hyperparameter_search(X_train, y_train, input_shape, epochs):
    param_space = {
        'num_layers': Integer(1, 3),
        'filters': Categorical(['[32]', '[64, 32]', '[128, 64, 32]']),
        'kernel_size': Categorical(['[5]', '[3, 5]', '[3, 3, 5]']),
        'stride': Categorical(['[1]', '[1, 2]', '[1, 2, 2]']),
        'pool_size': Categorical(['[2]', '[2, 2]', '[2, 2, 2]']),
        'activation': Categorical(['relu', 'tanh']),
        'learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
        'batch_size': Integer(16, 64)
    }

    regressor = PyTorchRegressorCV(input_shape, epochs=epochs)

    opt = BayesSearchCV(
        estimator=regressor,
        search_spaces=param_space,
        n_iter=3,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1
    )

    opt.fit(X_train, y_train)

    best_params = opt.best_params_
    best_params['filters'] = eval(best_params['filters'])
    best_params['kernel_size'] = eval(best_params['kernel_size'])
    best_params['stride'] = eval(best_params['stride'])
    best_params['pool_size'] = eval(best_params['pool_size'])

    return best_params

# Initialize or load previous results
def load_results(results_file):
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    else:
        return {"hyperparameter_trials": [], "best_mse": float('inf'), "best_hyperparameters": {}}

# Save updated results to JSON
def save_results(results, results_file):
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

# Save the current best model by overriding the previous one
def save_best_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"New best model saved to {path}")

def save_model(self, path='best_model.pth'):
    torch.save(self.model.state_dict(), path)

def load_model(self, path='best_model.pth'):
    self.model.load_state_dict(torch.load(path))
    self.model.to(self.device)
    return self.model

if __name__ == "__main__":

    epochs = 100

    # Paths and filenames
    results_file = "v2/Testing Expression Prediction/Hyperparameter Search/CNN_6_3_cross_validation/hyperparameter_results.json"
    runtime_model_path = "v2/Models/CNN_6_3_runtime.pth"
    model_path = 'v2/Models/CNN_6_3.pt'

    # Load all datasets from the directory
    files = glob.glob('v2/Data/Cross Validation/*.csv')
    file_data = {file.split('\\')[-1].split('.csv')[0]: load_features(file) for file in files}

    # Load previous results (or start fresh)
    results = load_results(results_file)

    all_mse_scores = []
    best_hyperparams = None
    best_mse = float('inf')

    # Perform Leave-One-Out Cross-Validation (LOOCV)
    for i, (test_key, (X_test, y_test)) in enumerate(file_data.items()):
        
        print(f"Fold {i + 1}: Test File, {test_key}")

        # Prepare training data for the current fold
        X_train_list = [X for key, (X, y) in file_data.items() if key != test_key]
        y_train_list = [y for key, (X, y) in file_data.items() if key != test_key]

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Define input shape for the model
        input_shape = (X_train.shape[0], X_train.shape[1], X_train.shape[2])

        # Perform hyperparameter search (for this fold)
        params = hyperparameter_search(X_train, y_train, input_shape, epochs)
        print(f"Best Hyperparameters for Fold {i + 1}: {params}")

        # Train the model with the best hyperparameters for this fold
        model = PyTorchRegressor(input_shape, params, epochs=epochs)
        model.fit(X_train, y_train)

        # Make predictions on the test dataset
        y_pred = model.predict(X_test)

        # Calculate MSE for this fold
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {i + 1}: MSE = {mse:.4f}")
        all_mse_scores.append(mse)

        # Save the hyperparameters, fold metrics, and MSE to results JSON
        trial_data = {
            "fold": i + 1,
            "hyperparameters": params,
            "mse": mse,
            "training_data_size": len(X_train),
            "test_data_size": len(X_test)
        }
        results["hyperparameter_trials"].append(trial_data)

        # Check if this fold has the best MSE so far
        if mse < best_mse:
            best_mse = mse
            best_hyperparams = params

            # Update best model metrics in the results JSON
            results["best_mse"] = best_mse
            results["best_hyperparameters"] = best_hyperparams

            # Save the best model by overriding the previous one
            save_best_model(model, runtime_model_path)

        # Save the updated results to the JSON file after each fold
        save_results(results)

    # Print final summary
    print(f"All MSE Scores: {all_mse_scores}")
    print(f"Best Hyperparameters: {best_hyperparams}")
    print(f"Best MSE: {best_mse}")

    # Train the final model with the best hyperparameters on all data
    X_all = np.concatenate([X for X, y in file_data.values()], axis=0)
    y_all = np.concatenate([y for X, y in file_data.values()], axis=0)

    input_shape = (X_all.shape[0], X_all.shape[1], X_all.shape[2])

    final_model = PyTorchRegressor(input_shape, best_hyperparams, epochs=epochs)
    final_model.fit(X_all, y_all)

    # Save the final model
    torch.save(final_model.model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
