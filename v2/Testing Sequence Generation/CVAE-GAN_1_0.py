import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import random

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_dataloader(df, batch_size):
    sequences = df['Promoter Sequence'].values
    expressions = df['Normalized Expression'].values
    
    x, y = [], []
    for seq in sequences:
        x.append(one_hot_encode_sequence(mask_sequence(seq)))
        y.append(one_hot_encode_sequence(seq))

    x = torch.stack(x)
    y = torch.stack(y)
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
    return torch.tensor([encoding[base.upper()] for base in padded_seq], dtype=torch.float32)

def mask_sequence(sequence, max_mask_length=20):
    seq_length = len(sequence)
    mask_length = random.randint(5, max_mask_length)
    start = random.randint(0, seq_length - mask_length)
    return sequence[:start] + 'N' * mask_length + sequence[start + mask_length:]

# CVAE Generator Model
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
        x = x.transpose(1, 2)
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mean, logvar)
        z_cond = torch.cat([z, condition], dim=1)
        recon = self.decoder(z_cond).view(-1, 150, 4)
        return recon, mean, logvar

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 150, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Match Conv1d input shape
        return self.model(x)

# Loss Functions
def compute_loss(generator, discriminator, x, condition, lambda_adv=0.1):
    # Reconstruction and KL loss
    recon, mean, logvar = generator(x, condition)
    reconstruction_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    # Adversarial loss
    real_pred = discriminator(x)
    fake_pred = discriminator(recon.detach())
    adv_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred)) + \
               F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))

    # Generator's adversarial loss (encourage fake sequences to be realistic)
    generator_adv_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

    # Total loss for the generator
    total_loss = reconstruction_loss + kl_loss + lambda_adv * generator_adv_loss
    return total_loss, adv_loss

# Training Functions
def train_step(generator, discriminator, x, condition, optimizer_g, optimizer_d):
    generator.train()
    discriminator.train()

    # Train Discriminator
    optimizer_d.zero_grad()
    _, adv_loss = compute_loss(generator, discriminator, x, condition)
    adv_loss.backward()
    optimizer_d.step()

    # Train Generator
    optimizer_g.zero_grad()
    total_loss, _ = compute_loss(generator, discriminator, x, condition)
    total_loss.backward()
    optimizer_g.step()

    return total_loss.item(), adv_loss.item()

def train_generator(generator, discriminator, optimizer_g, optimizer_d, train_loader, epochs):
    for epoch in range(epochs):
        total_loss, total_adv_loss = 0, 0
        for train_x, train_cond, _ in train_loader:
            loss, adv_loss = train_step(generator, discriminator, train_x, train_cond, optimizer_g, optimizer_d)
            total_loss += loss
            total_adv_loss += adv_loss
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Adversarial Loss: {total_adv_loss / len(train_loader):.4f}')

# Main Function
def main():
    # Hyperparameters
    batch_size = 32
    epochs = 10
    latent_dim = 16

    # Paths to Data
    path_to_train_data = 'v2/Data/Train Test/train_data.csv'
    path_to_test_data = 'v2/Data/Train Test/train_data.csv'

    # Load Data and Prepare Dataloaders
    train_df = load_data(path_to_train_data)
    train_loader = prepare_dataloader(train_df, batch_size)

    # Initialize Models and Optimizers
    generator = CVAE(latent_dim)
    discriminator = Discriminator()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    # Train Models
    train_generator(generator, discriminator, optimizer_g, optimizer_d, train_loader, epochs)

if __name__ == '__main__':
    main()
