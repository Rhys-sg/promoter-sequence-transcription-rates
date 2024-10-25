import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random

# One-hot encoding function
def one_hot_encode_sequence(seq, length=150):
    encoding = {
        'A': [1, 0, 0, 0],
        'T': [0, 1, 0, 0],
        'C': [0, 0, 1, 0],
        'G': [0, 0, 0, 1],
        'N': [0.25, 0.25, 0.25, 0.25],  # Masked section
        '0': [0, 0, 0, 0]  # Padding
    }
    padded_seq = seq.ljust(length, '0')
    encoded = np.array([encoding[base.upper()] for base in padded_seq], dtype=np.float32)
    return encoded

# Masking function
def mask_sequence(sequence, max_mask_length=20):
    seq_length = len(sequence)
    mask_length = random.randint(5, max_mask_length)
    start = random.randint(0, seq_length - mask_length)
    return sequence[:start] + 'N' * mask_length + sequence[start + mask_length:]

# Load and preprocess data
def load_and_preprocess_data(train_path, test_path, seq_length=150):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = np.stack(train_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length)))
    X_test = np.stack(test_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length)))

    y_train = train_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)
    y_test = test_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)

    return (torch.tensor(X_train), torch.tensor(y_train)), (torch.tensor(X_test), torch.tensor(y_test))

# Define CVAE using nn.Module
class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 150, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 150 * 4),
            nn.Sigmoid(),
            nn.Unflatten(1, (4, 150))
        )

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, condition):
        encoded = self.encoder(x)
        mean, logvar = encoded.chunk(2, dim=1)
        z = self.reparameterize(mean, logvar)
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond), mean, logvar

# Loss function
def compute_loss(model, x, condition):
    reconstruction, mean, logvar = model(x, condition)
    reconstruction_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_loss

# Training step
def train_step(model, optimizer, x, condition):
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, x, condition)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
def train_model(model, optimizer, train_loader, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for x, condition in train_loader:
            loss = train_step(model, optimizer, x, condition)
            total_loss += loss
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, condition in test_loader:
            loss = compute_loss(model, x, condition)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# Main function
def main():
    train_path = 'v2/Data/Train Test/train_data.csv'
    test_path = 'v2/Data/Train Test/test_data.csv'
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data(train_path, test_path)

    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    latent_dim = 16
    model = CVAE(latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, optimizer, train_loader, epochs=10)
    test_loss = evaluate_model(model, test_loader)
    print(f'Average Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()