import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(file_path, batch_size):
    df, scaler = load_data(file_path)

    X_sequence, X_expressions, X_start_idx, X_len_removed, y = preprocess_X_y(df)

    # Convert data to PyTorch Tensors
    X_sequence = torch.Tensor(X_sequence)
    X_expressions = torch.Tensor(X_expressions).unsqueeze(1)
    X_start_idx = torch.Tensor(X_start_idx).unsqueeze(1)
    X_len_removed = torch.Tensor(X_len_removed).unsqueeze(1)
    y = torch.Tensor(y)

    # Create dataset and dataloader
    dataset = TensorDataset(X_sequence, X_expressions, X_start_idx, X_len_removed, y)
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

def preprocess_X_y(df, num_augmentations=1, min_length=1, max_length=10):
    sequences, expressions = combine_columns(df)

    X_sequence = []
    X_expressions = []
    y = []
    X_start_idx = []
    X_len_removed = []

    for full_sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(min_length, max_length)
            masked_sequence, missing_element, start_idx = remove_section_get_features(full_sequence, len_removed)

            X_sequence.append(one_hot_encode_input(apply_padding(masked_sequence, 150)))
            X_expressions.append(expression)
            X_start_idx.append(start_idx)
            X_len_removed.append(len_removed)
            y.append(one_hot_encode_output(missing_element))

    return np.array(X_sequence), np.array(X_expressions), np.array(X_start_idx), np.array(X_len_removed), np.array(y)

def remove_section_get_features(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + 'N' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element, start_idx

def apply_padding(sequence, max_length):
    return '0' * (max_length - len(sequence)) + sequence

def one_hot_encode_input(sequence):
    mapping = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25],  # Ambiguous nucleotide
        '0': [0, 0, 0, 0]  # Padding
    }
    return [mapping[nucleotide.upper()] for nucleotide in sequence]

def one_hot_encode_output(sequence):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               '0': [0, 0, 0, 0]}  # Placeholder for padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

# Define CTGAN components (Generator and Discriminator)
class Generator(nn.Module):
    def __init__(self, sequence_length, latent_dim, expression_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_size=latent_dim + expression_dim, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 4)
    
    def forward(self, noise, expression, output_length):
        input_data = torch.cat([noise.unsqueeze(1), expression.unsqueeze(1)], dim=2)
        out, _ = self.lstm(input_data)
        out = self.fc(out[:, -output_length:, :]) # Dynamic sequence length
        return out


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

def initialize_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to initialize models
def initialize_models(sequence_length, latent_dim, expression_dim, device):
    generator = Generator(sequence_length, latent_dim, expression_dim)
    discriminator = Discriminator(sequence_length, expression_dim)
    return generator.to(device), discriminator.to(device)

# Function to insert the predicted sequence into the masked sequence
def insert_predicted_sequence(masked_sequence, predicted_sequence, start_index, length):
    predicted_sequence_str = decode_one_hot_sequences(predicted_sequence)[0]

    completed_sequence = (
        masked_sequence[:start_index] +
        predicted_sequence_str +
        masked_sequence[start_index + length:]
    )
    return completed_sequence

# Training the CTGAN
def train_ctgan(generator, discriminator, dataloader, num_epochs=100, latent_dim=100, expression_dim=1, lr=0.0002, device='cpu'):
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    criterion = nn.BCELoss()

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for i, (real_sequences, real_expressions, real_start_idx, real_len_removed, real_y) in enumerate(dataloader):
            batch_size = real_sequences.size(0)

            real_sequences = real_sequences.to(device)
            real_expressions = real_expressions.to(device)

            for j in range(batch_size):
                masked_sequence = real_sequences[j]
                start_index = real_start_idx[j]
                length = real_len_removed[j]

                # Generate noise and predict the missing sequence
                noise = torch.randn(1, latent_dim).to(device)  # Generate noise for one sample
                predicted_missing_sequence = generator(noise, real_expressions[j:j+1])

                # Insert the predicted sequence into the masked sequence
                completed_sequence = insert_predicted_sequence(masked_sequence, predicted_missing_sequence, start_index, length)

                # Convert completed_sequence to tensor for the discriminator
                completed_sequence_tensor = one_hot_encode_input(completed_sequence).to(device)

                # Train Discriminator
                real_labels = torch.ones(1, 1).to(device)
                fake_labels = torch.zeros(1, 1).to(device)

                real_prediction = discriminator(completed_sequence_tensor.unsqueeze(0), real_expressions[j:j+1])
                fake_prediction = discriminator(predicted_missing_sequence.detach(), real_expressions[j:j+1])

                d_loss_real = criterion(real_prediction, real_labels)
                d_loss_fake = criterion(fake_prediction, fake_labels)
                d_loss = (d_loss_real + d_loss_fake) / 2

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Train Generator
                fake_prediction = discriminator(predicted_missing_sequence, real_expressions[j:j+1])
                g_loss = criterion(fake_prediction, real_labels)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()

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
    generator, discriminator = initialize_models(sequence_length, latent_dim, expression_dim, device)

    # Train CTGAN
    train_ctgan(generator, discriminator, dataloader, num_epochs, latent_dim, expression_dim, device=device)

    # Save models
    save_model(generator, 'generator.pth')
    save_model(discriminator, 'discriminator.pth')

    # Evaluate generator
    real_expressions = [row['Normalized Observed log(TX/Txref)'] for _, row in pd.read_csv(file_path).iterrows()]
    generated_sequences = evaluate_generator(generator, real_expressions, latent_dim, device)
    decoded_sequences = decode_one_hot_sequences(generated_sequences)

    print("Generated Sequences:")
    for seq in decoded_sequences:
        print(seq)
