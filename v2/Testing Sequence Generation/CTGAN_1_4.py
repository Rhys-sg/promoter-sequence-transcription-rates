import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import random
import keras

def load_data(file_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    df['Normalized Observed log(TX/Txref)'] = scaler.fit_transform(df[['Observed log(TX/Txref)']])
    return df

def remove_section_get_features(sequence, mask_size=10):
    start = random.randint(0, len(sequence) - mask_size)
    return sequence[:start] + 'N' * mask_size + sequence[start + mask_size:], sequence[start:start + mask_size]

def one_hot_encode_sequence(seq, length=150):
    encoding = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0], 
                'G': [0, 0, 0, 1],
                'N': [0.25, 0.25, 0.25, 0.25],
                '0': [0, 0, 0, 0]}
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

class KerasModelWrapper(torch.nn.Module):
    def __init__(self, path_to_cnn):
        super(KerasModelWrapper, self).__init__()
        self.keras_model = keras.models.load_model(path_to_cnn)

    def forward(self, x, verbose=0):
        x_np = x.detach().cpu().numpy()
        preds = self.keras_model.predict(x_np, verbose=verbose)
        return torch.tensor(preds).to(x.device)

def train_ctgan(generator, discriminator, dataloader, cnn, epochs, lr, lambda_adversarial, lambda_cnn):
    criterion = nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (X, expr, y_real) in enumerate(dataloader):

            print(f"Epoch [{epoch+1}/{epochs}]  Batch [{i+1}/{len(dataloader)}]", end='\r')

            batch_size = X.size(0)

            # Train Discriminator
            y_fake = generator(X, expr)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            optimizer_d.zero_grad()
            real_loss = criterion(discriminator(y_real, expr), real_labels)
            fake_loss = criterion(discriminator(y_fake.detach(), expr), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_d.step()          

            # Train Generator with combined loss
            optimizer_g.zero_grad()
            adversarial_loss = criterion(discriminator(y_fake, expr), real_labels)
            cnn_loss = get_cnn_loss(cnn, X, y_fake, y_real, expr, batch_size)

            # Combine adversarial loss and CNN loss
            g_loss = (lambda_adversarial * adversarial_loss) + (lambda_cnn * cnn_loss)
            g_loss.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss aD: {adversarial_loss.item():.4f}, Loss cD: {cnn_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

def get_cnn_loss(cnn, X, y_fake, y_real, expr, batch_size):
    mse = nn.MSELoss()
    losses = []

    # Generate full sequences with infill and predict expression using CNN
    for i in range(batch_size):

        # Predict expression using the CNN
        pred_gen_expr = cnn(preprocess_cnn_input(X, y_fake, i)).item()

        # We should use the CNN-predicted expression for y_real here, but cuts time by half
        pred_real_expr = expr[i].item()

        # Calculate MSE loss between predicted and real expression
        loss = mse(torch.tensor([pred_gen_expr]), torch.tensor([pred_real_expr]))
        losses.append(loss)
    
    return torch.stack(losses).mean()

def preprocess_cnn_input(X, y, i):
    # Decode one-hot encoded sequences
    segment = decode_one_hot_sequence(y[i].argmax(dim=1).numpy())
    original_seq = decode_tensor_to_sequence(X[i])
    masked_seq, _ = remove_section_get_features(original_seq)

    # Reconstruct the full sequence by replacing masked segment with generated and real segments
    start = masked_seq.find('N' * y.size(1))
    infilled_seq = (
        masked_seq[:start] + segment + masked_seq[start + len(segment):]
    )

    # One-hot encode the infilled sequence for CNN prediction
    return one_hot_encode_sequence(infilled_seq).unsqueeze(0)

# Model Evaluation
def evaluate_generator(generator, cnn, test_loader):
    mse_loss = nn.MSELoss()
    total_loss = 0
    total_samples = 0

    # Iterate through the test data loader
    for X_batch, expr_batch, y_batch in test_loader:
        batch_size = X_batch.size(0)

        # Generate sequences and calculate CNN predictions
        y_fake = generator(X_batch, expr_batch)

        for i in range(batch_size):
            # Preprocess the input for the CNN
            cnn_input = preprocess_cnn_input(X_batch, y_fake, i)

            # Predict the expression using the CNN
            pred_expr = cnn(cnn_input).item()

            # Calculate the MSE loss with the real expression
            real_expr = expr_batch[i].item()
            loss = mse_loss(torch.tensor([pred_expr]), torch.tensor([real_expr]))

            total_loss += loss.item()
            total_samples += 1

    # Calculate and print the average MSE over the entire test set
    avg_mse = total_loss / total_samples
    print(f"Average MSE on Test Set: {avg_mse:.4f}")


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
        start = sequence.find('N' * mask_size)
        if start == -1:
            raise ValueError("No masked region ('N') found in the sequence.")
        
        # Convert the masked sequence to a tensor
        sequence_tensor = one_hot_encode_sequence(sequence).unsqueeze(0)
        expr_tensor = torch.tensor([expr], dtype=torch.float32).view(1, 1)

        # Generate infill using the generator
        generated_segment = generator(sequence_tensor, expr_tensor)
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
    adversarial_lambda = 1
    cnn_lambda = 10
    path_to_cnn = 'v2/Models/CNN_5_0.keras'
    path_to_data = 'v2/Data/combined/LaFleur_supp.csv'

    # Load Data and Prepare Dataloaders
    df = load_data(path_to_data)
    train_loader, test_loader = prepare_dataloader(df, batch_size)

    # Initialize Models
    generator = Generator()
    discriminator = Discriminator()
    cnn = KerasModelWrapper(path_to_cnn)

    # Train Models with Training DataLoader
    train_ctgan(generator, discriminator, train_loader, cnn, epochs, learning_rate, adversarial_lambda, cnn_lambda)

    # Save the trained models
    save_model(generator, 'generator.pth')
    save_model(discriminator, 'discriminator.pth')

    # Load the models
    load_model(generator, 'generator.pth')
    load_model(discriminator, 'discriminator.pth')

    # Evaluate the generator on the test set
    evaluate_generator(generator, cnn, test_loader)
    
    # Test Example
    sequences = ['TTTTCTATCTACGTACTTGACACTATTTCNNNNNNNNNNATTACCTTAGTTTGTACGTT']
    expressions = [0.5]

    # Generate infills
    infilled = generate_infills(generator, sequences, expressions)
    print("Infilled Sequences:", infilled)