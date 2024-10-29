import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random

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
    encoded = torch.tensor([encoding[base.upper()] for base in padded_seq], dtype=torch.float32)
    return encoded

def mask_sequence(sequence, max_mask_length=20):
    seq_length = len(sequence)
    mask_length = random.randint(5, max_mask_length)
    start = random.randint(0, seq_length - mask_length)
    masked_seq = sequence[:start] + 'N' * mask_length + sequence[start + mask_length:]
    return masked_seq

def load_and_preprocess_data(train_path, test_path, seq_length=150):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = torch.stack([one_hot_encode_sequence(mask_sequence(seq), seq_length) 
                           for seq in train_data['Promoter Sequence']])
    X_test = torch.stack([one_hot_encode_sequence(mask_sequence(seq), seq_length) 
                          for seq in test_data['Promoter Sequence']])

    y_train = torch.tensor(train_data['Normalized Expression'].values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(test_data['Normalized Expression'].values, dtype=torch.float32).view(-1, 1)

    return (X_train, y_train), (X_test, y_test)

class CVAE(nn.Module):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 150, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)  # Outputs mean and logvar
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 150 * 4),
            nn.Sigmoid()
        )

    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean)
        return eps * torch.exp(0.5 * logvar) + mean

    def forward(self, x, condition):
        # Transpose to match Conv1d expected input shape: (batch_size, channels, sequence_length)
        x = x.transpose(1, 2)
        
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mean, logvar)
        z_cond = torch.cat([z, condition], dim=1)
        recon = self.decoder(z_cond).view(-1, 150, 4)
        return recon, mean, logvar

def compute_loss(model, x, condition):
    recon, mean, logvar = model(x, condition)
    reconstruction_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + kl_loss

def train_step(model, x, condition, optimizer):
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, x, condition)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_model(model, optimizer, train_dataset, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for train_x, train_cond in train_dataset:
            loss = train_step(model, train_x, train_cond, optimizer)
            total_loss += loss
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataset):.4f}')

def evaluate_model(model, test_dataset):
    model.eval()
    losses = []
    with torch.no_grad():
        for test_x, test_cond in test_dataset:
            loss = compute_loss(model, test_x, test_cond)
            losses.append(loss.item())
    return np.mean(losses)

def main():
    train_path = 'v2/Data/Train Test/train_data.csv'
    test_path = 'v2/Data/Train Test/test_data.csv'
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data(train_path, test_path)

    latent_dim = 16
    model = CVAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32
    train_dataset = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    train_model(model, optimizer, train_dataset, epochs=10)

    test_loss = evaluate_model(model, test_dataset)
    print(f'Average Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()
