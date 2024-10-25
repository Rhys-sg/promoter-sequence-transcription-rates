import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
import numpy as np
import pandas as pd
import random

# === Utility Functions ===
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
    masked_seq = sequence[:start] + 'N' * mask_length + sequence[start + mask_length:]
    return masked_seq

def load_and_preprocess_data(train_path, test_path, seq_length=150):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Apply masking and one-hot encoding
    X_train = np.stack(train_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length)))
    X_test = np.stack(test_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length)))

    y_train = train_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)
    y_test = test_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)

    return (X_train, y_train), (X_test, y_test)

# === CVAE Model ===
class CVAE(nn.Module):
    def __init__(self, latent_dim=16):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: Conv1d followed by activation functions
        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, 3),  # Input: (batch_size, channels=4, seq_length=150)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 148, 64),  # Adjust for flattened size
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim)  # Outputs mean and log-variance
        )

        # Decoder: Linear layers with activations and reshaping
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 150 * 4),
            nn.Sigmoid(),
            nn.Unflatten(1, (4, 150))  # Reshape to (batch_size, channels=4, seq_length=150)
        )

    def reparameterize(self, mean, logvar):
        """Reparameterization trick to sample latent variable."""
        eps = torch.randn_like(mean)
        return eps * torch.exp(logvar * 0.5) + mean

    def forward(self, x, expr):
        # Transpose input from (batch_size, seq_length, channels) to (batch_size, channels, seq_length)
        x = x.permute(0, 2, 1)

        # Encode input to latent space
        encoded = self.encoder(x)
        mean, logvar = torch.chunk(encoded, 2, dim=1)  # Split into mean and log-variance

        # Reparameterize to get latent variable z
        z = self.reparameterize(mean, logvar)

        # Concatenate z with expression level
        z_cond = torch.cat([z, expr], dim=1)

        # Decode to reconstruct the original input
        reconstructed = self.decoder(z_cond)
        return reconstructed, mean, logvar

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(150 * 4 + 1, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, y, expr):
        y_flat = y.view(y.size(0), -1)
        combined = torch.cat([y_flat, expr], dim=1)
        return self.net(combined)

# === Training Functions ===
def compute_reconstruction_loss(x, reconstructed):
    return nn.MSELoss()(x, reconstructed)

def compute_kl_loss(mean, logvar):
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

def compute_auxiliary_loss(cnn, generated, expr):
    preds = cnn(generated).view(-1, 1)
    return nn.MSELoss()(preds, expr)

def train_cvae_gan(generator, discriminator, cnn, train_loader, epochs, lr, lambda_adv, lambda_aux):
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for X_batch, expr_batch in train_loader:
            batch_size = X_batch.size(0)

            # === Train Discriminator ===
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            reconstructed, _, _ = generator(X_batch, expr_batch)
            real_loss = nn.BCELoss()(discriminator(X_batch, expr_batch), real_labels)
            fake_loss = nn.BCELoss()(discriminator(reconstructed.detach(), expr_batch), fake_labels)
            d_loss = real_loss + fake_loss

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # === Train Generator ===
            reconstructed, mean, logvar = generator(X_batch, expr_batch)
            adv_loss = nn.BCELoss()(discriminator(reconstructed, expr_batch), real_labels)
            recon_loss = compute_reconstruction_loss(X_batch, reconstructed)
            kl_loss = compute_kl_loss(mean, logvar)
            aux_loss = compute_auxiliary_loss(cnn, reconstructed, expr_batch)

            g_loss = recon_loss + kl_loss + lambda_adv * adv_loss + lambda_aux * aux_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# === Main Function ===
def main():
    train_path = 'v2/Data/Train Test/train_data.csv'
    test_path = 'v2/Data/Train Test/test_data.csv'
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data(train_path, test_path)

    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    generator = CVAE(latent_dim=16)
    discriminator = Discriminator()
    cnn = tf.keras.models.load_model('v2/Models/CNN_5_0.keras')

    epochs = 10
    learning_rate = 0.0002
    lambda_adv = 1.0
    lambda_aux = 10.0

    train_cvae_gan(generator, discriminator, cnn, train_loader, epochs, learning_rate, lambda_adv, lambda_aux)

if __name__ == '__main__':
    main()
