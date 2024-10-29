import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
import random

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filepath):
    return pd.read_csv(filepath)

def mask_data(df, num_masks=1, num_inserts=1, min_mask=1, max_mask=10):
        
    original_sequences = df['Promoter Sequence']
    masked_sequences = []
    random_infilled_sequences = []

    for sequence in original_sequences:
        for _ in range(num_masks):
            mask_length = random.randint(min_mask, max_mask)
            mask_start = random.randint(0, len(sequence) - mask_length)

            # Mask the sequence
            masked_seq = sequence[:mask_start] + 'N' * mask_length + sequence[mask_start + mask_length:]

            # Generate multiple random infills for this masked sequence
            for _ in range(num_inserts):
                random_infill = ''.join(random.choices('ATCG', k=mask_length))
                random_infilled_seq = masked_seq[:mask_start] + random_infill + masked_seq[mask_start + len(random_infill):]

                # Collect results
                masked_sequences.append(masked_seq)
                random_infilled_sequences.append(random_infilled_seq)

    # Construct the new DataFrame with required columns
    new_df = pd.DataFrame({
        'Original Promoter Sequence': original_sequences.repeat(num_masks * num_inserts).reset_index(drop=True),
        'Masked Promoter Sequence': masked_sequences,
        'Random Infilled Promoter Sequence': random_infilled_sequences
    })

    return new_df

def preprocess_data(df, cnn, device):

    def one_hot_sequence(seq, length=150):
        seq = seq.ljust(length, '0')
        return np.eye(5)[['ACGTN0'.index(nt) for nt in seq]]

    # Apply one-hot encoding and batch the sequences
    sequences = df['Promoter Sequence'].apply(lambda x: one_hot_sequence(x))
    sequences = torch.tensor(np.stack(sequences), dtype=torch.float32).to(device)

    # Predict expression using the CNN in batches (disable gradients)
    with torch.no_grad():
        expression = cnn(sequences).squeeze(1)

    return sequences, expression

class CVAE(nn.Module):
    def __init__(self, seq_length, latent_size, class_size):
        super(CVAE, self).__init__()
        self.seq_length = seq_length * 5  # One-hot encoded sequence (5 channels)
        self.class_size = class_size

        # Encoder
        self.fc1 = nn.Linear(self.seq_length + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)  # Mean
        self.fc22 = nn.Linear(400, latent_size)  # Log-variance

        # Decoder
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, self.seq_length)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c):
        inputs = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        inputs = torch.cat([z, c], 1)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.seq_length), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

def loss_function(recon_x, x, mu, logvar, cnn, context_expression):
    generated_expression = cnn(recon_x.view(-1, 5, 150)).squeeze(1)
    AUX = F.mse_loss(generated_expression, context_expression)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 750), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD + AUX

def train(epoch, model, cnn, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, expression) in enumerate(train_loader):
        data, expression = data.to(device), expression.to(device).unsqueeze(1)
        recon_batch, mu, logvar = model(data, expression)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar, cnn, expression)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item() / len(data):.6f}', end='\r')
    print(f'\n====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch, model, cnn, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, expression) in enumerate(test_loader):
            data, expression = data.to(device), expression.to(device).unsqueeze(1)
            recon_batch, mu, logvar = model(data, expression, cnn)
            test_loss += loss_function(recon_batch, data, mu, logvar, cnn, expression).item()
    print(f'====> Test set loss: {test_loss / len(test_loader.dataset):.4f}')

def main():

    # Set seed for reproducibility
    seed = 42
    random.seed(seed)

    # Paths to Data and Pre-trained CNN
    path_to_train_data = 'v2/Data/Train Test/train_data.csv'
    path_to_test_data = 'v2/Data/Train Test/train_data.csv'
    path_to_cnn = 'v2/Models/CNN_6_2.pt'

    # Set up device
    device = get_device()

    # Initialize model, optimizer
    latent_size = 20
    cnn = torch.jit.load(f'{path_to_cnn}.pt')
    model = CVAE(150, latent_size, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load data
    train_df = load_data(path_to_train_data)
    test_df = load_data(path_to_test_data)

    # Augment/mask data
    augmented_train_df = mask_data(train_df, num_masks=3, num_inserts=1, min_mask=1, max_mask=10)
    augmented_test_df = mask_data(test_df, num_masks=3, num_inserts=1, min_mask=1, max_mask=10)

    # Preprocess data
    train_sequences, train_expression = preprocess_data(augmented_train_df, cnn, device)
    test_sequences, test_expression = preprocess_data(augmented_test_df, cnn, device)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_sequences, train_expression),
        batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_sequences, test_expression),
        batch_size=64, shuffle=False
    )

    # Train and test the model
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, model, cnn, train_loader, optimizer, device)
        test(epoch, model, cnn, test_loader, device)

if __name__ == "__main__":
    main()
