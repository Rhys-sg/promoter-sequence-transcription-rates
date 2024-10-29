import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import random

def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    df['Normalized Observed log(TX/Txref)'] = scaler.fit_transform(df[['Observed log(TX/Txref)']])
    return df

def remove_section_get_features(sequence, mask_size=10):
    start = random.randint(0, len(sequence) - mask_size)
    return sequence[:start] + 'N' * mask_size + sequence[start + mask_size:], sequence[start:start + mask_size]

def one_hot_encode_sequence(seq, length=150):
    encoding = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 
                'G': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25], '0': [0, 0, 0, 0]}
    padded_seq = seq.ljust(length, '0')
    return torch.tensor([encoding[base.upper()] for base in padded_seq], dtype=torch.float32)

from sklearn.model_selection import train_test_split

def prepare_dataloader(df, batch_size=64, test_size=0.01):
    sequences = df['Promoter Sequence'].values
    expressions = df['Normalized Observed log(TX/Txref)'].values
    
    X, y = [], []

    for seq in sequences:
        masked_seq, target = remove_section_get_features(seq)
        X.append(one_hot_encode_sequence(masked_seq))
        y.append(one_hot_encode_sequence(target, length=10))

    X = torch.stack(X)  # Shape: (num_samples, 150, 4)
    y = torch.stack(y)  # Shape: (num_samples, 10, 4)
    expressions = torch.tensor(expressions, dtype=torch.float32).view(-1, 1)

    # Train-Test Split
    X_train, X_test, expr_train, expr_test, y_train, y_test = train_test_split(
        X, expressions, y, test_size=test_size, random_state=42
    )

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, expr_train, y_train)
    test_dataset = TensorDataset(X_test, expr_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class Generator(nn.Module):
    def __init__(self, input_seq_dim=150 * 4, expr_dim=1, hidden_dim=512, output_dim=10 * 4):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_seq_dim + expr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, expr):
        x_flat = x.view(x.size(0), -1)  # (batch_size, 150 * 4)
        combined = torch.cat((x_flat, expr), dim=1)  # (batch_size, 150 * 4 + 1)
        return self.net(combined).view(-1, 10, 4)  # Output reshaped to (batch_size, 10, 4)

class Discriminator(nn.Module):
    def __init__(self, input_seq_dim=10 * 4, expr_dim=1, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_seq_dim + expr_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, y, expr):
        y_flat = y.view(y.size(0), -1)  # (batch_size, 10 * 4)
        combined = torch.cat((y_flat, expr), dim=1)  # (batch_size, 41)
        return self.net(combined)  # Output a probability (real vs fake)

def train_ctgan(generator, discriminator, dataloader, epochs, lr):
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, expr, y_real in dataloader:
            # Train Discriminator
            y_fake = generator(X, expr)
            real_labels = torch.ones(X.size(0), 1)
            fake_labels = torch.zeros(X.size(0), 1)

            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(y_real, expr), real_labels)
            fake_loss = criterion(discriminator(y_fake.detach(), expr), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            g_loss = criterion(discriminator(y_fake, expr), real_labels)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

# Model Evaluation
def evaluate_generator(generator, X_tensor, expr):
    sequence = decode_tensor_to_sequence(X_tensor.squeeze(0))
    masked_seq, target_segment = remove_section_get_features(sequence)
    start = sequence.find('N' * len(target_segment))
    masked_seq_tensor = one_hot_encode_sequence(masked_seq).unsqueeze(0)
    expr_tensor = expr.view(1, 1)
    generated_seq = generator(masked_seq_tensor, expr_tensor)
    predicted_infill = decode_one_hot_sequence(generated_seq.argmax(dim=2).squeeze().numpy())

    reconstructed_seq = (
        sequence[:start] + predicted_infill + sequence[start + len(target_segment):]
    )

    return reconstructed_seq

def decode_tensor_to_sequence(tensor):
    """Convert a one-hot encoded tensor back into a string sequence."""
    decoding = ['A', 'T', 'C', 'G']
    sequence = ''.join(decoding[base.argmax().item()] for base in tensor)
    return sequence

def decode_one_hot_sequence(encoded_seq):
    decoding = ['A', 'T', 'C', 'G', 'N']
    return ''.join([decoding[i] for i in encoded_seq])

def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")


def generate_infills(generator, sequences, expressions, mask_size=10):
    infilled_sequences = []
    
    for sequence, expr in zip(sequences, expressions):
        # Locate the masked section (assumes 'N' is used for masked regions)
        start = sequence.find('N' * mask_size)
        if start == -1:
            raise ValueError("No masked region ('N') found in the sequence.")
        
        # Convert the masked sequence to a tensor
        sequence_tensor = one_hot_encode_sequence(sequence).unsqueeze(0)  # Shape: (1, 150, 4)
        expr_tensor = torch.tensor([expr], dtype=torch.float32).view(1, 1)  # Shape: (1, 1)

        # Generate infill using the generator
        generated_segment = generator(sequence_tensor, expr_tensor)  # Shape: (1, 10, 4)
        predicted_infill = decode_one_hot_sequence(generated_segment.argmax(dim=2).squeeze().numpy())
        
        # Reconstruct the full sequence
        infilled_sequence = (
            sequence[:start] + predicted_infill + sequence[start + mask_size:]
        )
        infilled_sequences.append(infilled_sequence)
    
    return infilled_sequences


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    epochs = 5
    learning_rate = 0.0002

    # Load Data and Prepare Dataloaders
    df = load_data('v2/Data/combined/LaFleur_supp.csv')
    train_loader, test_loader = prepare_dataloader(df, batch_size)

    # Initialize Models
    generator = Generator()
    discriminator = Discriminator()

    # Train Models with Training DataLoader
    train_ctgan(generator, discriminator, train_loader, epochs, learning_rate)

    # Save the trained models
    save_model(generator, 'generator.pth')
    save_model(discriminator, 'discriminator.pth')

    # Load the models
    load_model(generator, 'generator.pth')
    load_model(discriminator, 'discriminator.pth')

    # Evaluate with Test DataLoader
    for X_test, expr_test, y_test in test_loader:
        generated_seq = evaluate_generator(generator, X_test, expr_test)
        decoded_seq = decode_one_hot_sequence(generated_seq[0].numpy())
        print("Generated Sequence:", decoded_seq)

    
    # Test Example
    sequences = ['TTTTCTATCTACGTACTTGACACTATTTCNNNNNNNNNNATTACCTTAGTTTGTACGTT']
    expressions = [0.5]

    # Generate infills
    infilled = generate_infills(generator, sequences, expressions)
    print("Infilled Sequences:", infilled)