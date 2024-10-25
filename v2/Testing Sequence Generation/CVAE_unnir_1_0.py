import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np

def pad_sequence(seq, length=150):
    return seq + 'N' * (length - len(seq))

def one_hot_sequence(seq):
    mapping = {'A': [1, 0, 0, 0, 0], 'C': [0, 1, 0, 0, 0], 
               'G': [0, 0, 1, 0, 0], 'T': [0, 0, 0, 1, 0], 
               'N': [0, 0, 0, 0, 1]}
    return np.array([mapping[nt] for nt in seq])

def load_data(filepath):
    data = pd.read_csv(filepath)
    sequences = data['Promoter Sequence'].apply(lambda x: pad_sequence(x, 150))
    sequences = np.array([one_hot_sequence(seq) for seq in sequences])
    expression = data['Normalized Expression'].values
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(expression, dtype=torch.float32)

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

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 750), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch, model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, expression) in enumerate(train_loader):
        data, expression = data.to(device), expression.to(device).unsqueeze(1)
        recon_batch, mu, logvar = model(data, expression)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item() / len(data):.6f}')
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(epoch, model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, expression) in enumerate(test_loader):
            data, expression = data.to(device), expression.to(device).unsqueeze(1)
            recon_batch, mu, logvar = model(data, expression)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    print(f'====> Test set loss: {test_loss / len(test_loader.dataset):.4f}')

# Main function
def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_sequences, train_expression = load_data('v2/Data/Train Test/train_data.csv')
    test_sequences, test_expression = load_data('v2/Data/Train Test/test_data.csv')

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_sequences, train_expression),
        batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_sequences, test_expression),
        batch_size=64, shuffle=False
    )

    # Initialize model, optimizer
    latent_size = 20
    model = CVAE(150, latent_size, 1).to(device)  # 150 nucleotides, 5 channels
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train and test the model
    epochs = 10
    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, optimizer, device)
        test(epoch, model, test_loader, device)

        # # Generate new sequences
        # with torch.no_grad():
        #     expression_values = torch.linspace(0, 1, steps=10).unsqueeze(1).to(device)
        #     sample = torch.randn(10, latent_size).to(device)
        #     generated_sequences = model.decode(sample, expression_values).cpu()
        #     print("Generated Sequences:", generated_sequences)

if __name__ == "__main__":
    main()
