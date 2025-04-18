import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import keras


def load_and_preprocess_data(file_path, batch_size):
    df, scaler = load_data(file_path)

    X_sequence, X_expressions, y = preprocess_X_y(df)

    X_sequence_train, X_sequence_test, X_expressions_train, X_expressions_test, y_train, y_test = train_test_split(
        X_sequence, X_expressions, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch Tensors
    X_sequence_train = torch.Tensor(X_sequence_train)
    X_expressions_train = torch.Tensor(X_expressions_train).unsqueeze(1)
    y_train = torch.Tensor(y_train)

    # Create dataset and dataloader
    dataset = TensorDataset(X_sequence_train, X_expressions_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    df['Normalized Observed log(TX/Txref)'] = scaler.fit_transform(df[['Observed log(TX/Txref)']])
    return df, scaler

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_X_y(df, num_augmentations=1):
    sequences, expressions = combine_columns(df)

    X_sequence = []
    X_expressions = []
    y = []

    for full_sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(1, 10)
            masked_sequence, missing_element = remove_section_get_features(full_sequence, len_removed)

            X_sequence.append(one_hot_encode_input(apply_padding(masked_sequence, 150)))
            X_expressions.append(expression)
            y.append(one_hot_encode_output(apply_padding(full_sequence, 150)))

    return np.array(X_sequence), np.array(X_expressions), np.array(y)

def remove_section_get_features(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + 'N' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element

def apply_padding(sequence, max_length):
    return '0' * (max_length - len(sequence)) + sequence

def one_hot_encode_input(sequence):
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25],  # Ambiguous nucleotide, equal contribution from all
        '0': [0, 0, 0, 0]  # Padding
    }
    return [mapping[nucleotide.upper()] for nucleotide in sequence]

def one_hot_encode_output(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               '0': [0, 0, 0, 0]}  # Padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

# Define CTGAN components (Generator and Discriminator)
    
class Generator(nn.Module):
    def __init__(self, sequence_length, latent_dim, expression_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim + expression_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, sequence_length * 4) # Output size is one-hot encoded sequence (A, T, C, G)

    def forward(self, noise, expression):
        x = torch.cat([noise, expression], dim=1)
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        x = x.view(x.size(0), -1, 4)  # Reshape to (batch_size, sequence_length, 4)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, sequence_length, expression_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(sequence_length * 4 + expression_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, sequence, expression):
        sequence_flat = sequence.view(sequence.size(0), -1)
        x = torch.cat([sequence_flat, expression], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

class KerasModelWrapper(torch.nn.Module):
    def __init__(self, keras_model):
        super(KerasModelWrapper, self).__init__()
        self.keras_model = keras_model

    def forward(self, x, verbose=0):
        x_np = x.detach().cpu().numpy()
        preds = self.keras_model.predict(x_np, verbose=verbose)
        return torch.tensor(preds).to(x.device)
    
def initialize_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to initialize models
def initialize_models(sequence_length, latent_dim, expression_dim, device, path_to_cnn):
    generator = Generator(sequence_length, latent_dim, expression_dim).to(device)
    discriminator = Discriminator(sequence_length, expression_dim).to(device)
    cnn = KerasModelWrapper(keras.models.load_model(path_to_cnn)).to(device)
    return generator, discriminator, cnn

# Training the CTGAN
def train_ctgan(generator, discriminator, cnn, dataloader, num_epochs=100, latent_dim=100, expression_dim=1, lr=0.0002, device='cpu'):
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    criterion = nn.BCELoss()
    expression_loss = nn.MSELoss()

    # Set models to training mode, except for the CNN
    generator.train()
    discriminator.train()
    cnn.eval()  

    for epoch in range(num_epochs):
        for i, (real_sequences, real_expressions) in enumerate(dataloader):
            batch_size = real_sequences.size(0)
            real_sequences = real_sequences.to(device)
            real_expressions = real_expressions.to(device)

            # Train Discriminator
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_sequences = generator(noise, real_expressions)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            real_predictions = discriminator(real_sequences, real_expressions)
            fake_predictions = discriminator(fake_sequences.detach(), real_expressions)

            d_loss_real = criterion(real_predictions, real_labels)
            d_loss_fake = criterion(fake_predictions, fake_labels)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            fake_predictions = discriminator(fake_sequences, real_expressions)
            g_loss = criterion(fake_predictions, real_labels)

            # Use the pre-trained CNN to get predictions
            with torch.no_grad():
                g_cnn_loss = expression_loss(cnn(fake_sequences), real_expressions)

            total_g_loss = g_loss + g_cnn_loss

            g_optimizer.zero_grad()
            total_g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | CNN Loss: {g_cnn_loss.item():.4f}")

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")

# Evaluating the Generator (Sampling and comparing with real sequences)
def evaluate_generator(generator, real_expressions, latent_dim=100, device='cpu'):
    generator.eval()

    # Convert the list of expressions to a tensor
    real_expressions_tensor = torch.tensor(real_expressions).float().to(device).unsqueeze(1)

    noise = torch.randn(real_expressions_tensor.size(0), latent_dim).to(device)
    generated_sequences = generator(noise, real_expressions_tensor)
    return generated_sequences

def decode_one_hot_sequences(one_hot_sequences):
    nucleotide_mapping = {
        (1, 0, 0, 0): 'A',
        (0, 1, 0, 0): 'T',
        (0, 0, 1, 0): 'C',
        (0, 0, 0, 1): 'G',
        (0, 0, 0, 0): '',  # Padding
    }
    sequences = []
    for one_hot_sequence in one_hot_sequences:
        # Detach the tensor and convert to NumPy, then round
        rounded_sequence = np.round(one_hot_sequence.detach().numpy()).astype(int)
        decoded_sequence = ''.join(nucleotide_mapping.get(tuple(nucleotide), '') for nucleotide in rounded_sequence)
        sequences.append(decoded_sequence)
    return sequences


if __name__ == '__main__':
    # Hyperparameters
    sequence_length = 150
    latent_dim = 100
    expression_dim = 1
    batch_size = 64
    num_epochs = 100
    lr = 0.0002

    # Load and preprocess data
    file_path = 'v2/Data/combined/LaFleur_supp.csv'
    dataloader = load_and_preprocess_data(file_path, batch_size)

    # Initialize models
    device = initialize_device()
    generator, discriminator, cnn = initialize_models(sequence_length, latent_dim, expression_dim, device, 'v2/Models/CNN_5_0.keras')

    # Train the models
    train_ctgan(generator, discriminator, cnn, dataloader, num_epochs=num_epochs, latent_dim=latent_dim, expression_dim=expression_dim, lr=lr, device=device)

    # Save the trained models
    save_model(generator, 'generator.pth')
    save_model(discriminator, 'discriminator.pth')

    # Load the models
    load_model(generator, 'generator.pth')
    load_model(discriminator, 'discriminator.pth')

    # Values to evaluate the Generator
    sequences = ['TTTTCTATCTACGTACTTGACACTATTTC______________ATT__________ACCTTAGTTTGTACGTT']
    expressions = [0.5]

    # Evaluate the Generator
    generated_sequences = evaluate_generator(generator, expressions, latent_dim=latent_dim, device=device)
    print("Generated Sequences: ", generated_sequences)
    print("Decoded Sequences: ", decode_one_hot_sequences(generated_sequences))
