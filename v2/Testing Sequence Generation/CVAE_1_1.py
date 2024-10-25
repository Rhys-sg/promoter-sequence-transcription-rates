import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_dataloader(df, batch_size=64, test_size=0.01):
    sequences = df['Promoter Sequence'].values
    expressions = df['Normalized Expression'].values
    
    x, y = [], []

    for seq in sequences:
        x.append(one_hot_encode_sequence(mask_sequence(seq)))
        y.append(one_hot_encode_sequence(seq))

    x = torch.stack(x)  # Shape: (num_samples, 150, 4)
    y = torch.stack(y)  # Shape: (num_samples, 10, 4)
    expressions = torch.tensor(expressions, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(x, expressions, y)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

def train_model(model, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for train_x, train_cond, y in train_loader:
            loss = train_step(model, train_x, train_cond, optimizer)
            total_loss += loss
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for test_x, test_cond, y in test_loader:
            loss = compute_loss(model, test_x, test_cond)
            losses.append(loss.item())
    return np.mean(losses)

def main():
    # Hyperparameters
    batch_size = 32
    epochs = 10

    # Paths to Data
    path_to_train_data = 'v2/Data/Train Test/train_data.csv'
    path_to_test_data = 'v2/Data/Train Test/train_data.csv'

    # Load Data and Prepare Dataloaders
    train_df = load_data(path_to_train_data)
    test_df = load_data(path_to_test_data)

    train_loader = prepare_dataloader(train_df, batch_size)
    test_loader = prepare_dataloader(test_df, batch_size)


    latent_dim = 16
    model = CVAE(latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_model(model, optimizer, train_loader, epochs)

    test_loss = evaluate_model(model, test_loader)
    print(f'Average Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()
