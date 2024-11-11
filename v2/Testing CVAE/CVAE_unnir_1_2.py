import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
import numpy as np
import random
import keras
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_features(file_path, num_masks, min_mask_len, max_mask_len):
    df = pd.read_csv(file_path)
    df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(df[['Observed log(TX/Txref)']].abs())
    X = preprocess_sequences(df[['Promoter Sequence']].astype(str).agg(''.join, axis=1), num_masks, min_mask_len, max_mask_len)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_sequences(X, num_masks, min_mask_len, max_mask_len, max_length=150):
    masked_sequences = mask_sequences(X, num_masks, min_mask_len, max_mask_len)
    padded_sequences = [seq.zfill(max_length) for seq in X]
    return np.array([one_hot_sequence(seq) for seq in padded_sequences])

def mask_sequences(sequence, num_masks, min_mask_len, max_mask_len):
    mask_sequences = []
    for _ in range(num_masks):
        mask_len = random.randint(min_mask_len, max_mask_len)
        mask_start = random.randint(0, len(sequence) - mask_len)
        mask = 'N' * mask_len
        masked_sequence = sequence[:mask_start] + mask + sequence[mask_start + mask_len:]
        mask_sequences.append(masked_sequence)
    
    return mask_sequences
     
def one_hot_sequence(seq):
    mapping = {'A': [1, 0, 0, 0],
               'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'T': [0, 0, 0, 1],
               'N': [0.25, 0.25, 0.25, 0.25], # Ambiguous nucleotide mask
               '0': [0, 0, 0, 0]} # Padding
    return np.array([mapping[nucleotide.upper()] for nucleotide in seq])

class CVAE(nn.Module):
    def __init__(self, seq_length, latent_size, class_size):
        super(CVAE, self).__init__()
        self.seq_length = seq_length * 4  # One-hot encoded sequence (4 channels)
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
    
class KerasModelWrapper(torch.nn.Module):
    def __init__(self, path_to_cnn):
        super(KerasModelWrapper, self).__init__()
        self.keras_model = keras.models.load_model(path_to_cnn)

    def forward(self, x, verbose=0):
        x_np = x.detach().cpu().numpy()
        preds = self.keras_model.predict(x_np, verbose=verbose)
        return torch.tensor(preds).to(x.device)

def loss_function(recon_x, x, mu, logvar, cnn, context_expression):
    generated_expression = cnn(preprocess_for_cnn(x, recon_x.view(-1, 150, 4))).squeeze(1)
    AUX = F.mse_loss(generated_expression, context_expression.squeeze())
    BCE = F.binary_cross_entropy(recon_x.view(-1, 600), x.view(-1, 600), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD + AUX

def preprocess_for_cnn(x, recon):
    mask_value = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32, device=x.device)
    recon_one_hot = F.one_hot(recon.argmax(dim=-1), num_classes=4).float()
    final_sequence = x.clone()
    for i in range(x.shape[0]):
        mask = torch.all(x[i] == mask_value, dim=-1)
        final_sequence[i][mask] = recon_one_hot[i][mask]

    return final_sequence

def fit_model(epochs,
              model,
              cnn,
              path_to_summary,
              train_loader,
              test_loader,
              optimizer,
              device,
              early_stopping_patience,
              early_stopping_min_delta
    ):
    
    writer = SummaryWriter(path_to_summary)
    train_losses = [0] * epochs
    test_losses = [0] * epochs
    best_test_loss = float('inf')
    patience_counter = 0

    # Train and test the model
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model, cnn, train_loader, optimizer, device)
        test_loss = test(epoch, model, cnn, test_loader, device)
        
        # Check for improvement in test loss
        if best_test_loss - test_loss > early_stopping_min_delta:
            best_test_loss = test_loss
            patience_counter = 0
            improvement_status = ""
        else:
            patience_counter += 1
            improvement_status = f"| No improvement in test loss for {patience_counter} epoch(s)"
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} {improvement_status}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)

        train_losses[epoch-1] = train_loss
        test_losses[epoch-1] = test_loss

        # Check if patience has been exceeded
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    writer.close()

    return train_losses, test_losses

def train(epoch, model, cnn, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, expression) in enumerate(train_loader):
        data, expression = data.to(device), expression.to(device).unsqueeze(1)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, expression)
        loss = loss_function(recon_batch, data, mu, logvar, cnn, expression)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        train_loss += loss.item()
        optimizer.step()
        print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | Train Loss: {loss.item() / len(data):.6f}', end='\r')

    return train_loss / len(train_loader.dataset)

def test(epoch, model, cnn, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, expression in test_loader:
            data, expression = data.to(device), expression.to(device).unsqueeze(1)
            recon_batch, mu, logvar = model(data, expression)
            test_loss += loss_function(recon_batch, data, mu, logvar, cnn, expression).item()
    return test_loss / len(test_loader.dataset)

def plot_losses(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses over Epochs')
    plt.legend()
    plt.show()

def save_model(model, path):
    model_scripted = torch.jit.script(model) 
    model_scripted.save(path)

def load_model(path):
    return torch.jit.load(path)

def decode_one_hot(encoded_seq):
    mapping = {(1, 0, 0, 0) : 'A',
               (0, 1, 0, 0) : 'C',
               (0, 0, 1, 0) : 'G',
               (0, 0, 0, 1) : 'T',
               (0, 0, 0, 0) : ''} # Padding
    return [mapping[tuple(nucleotide)] for nucleotide in encoded_seq]

def generate_infills(model, cnn, sequences, expressions):
    
    # Convert sequences to one-hot encoding, and convert to tensor
    one_hot_sequences = [one_hot_sequence(seq) for seq in sequences]
    one_hot_sequences_tensor = torch.tensor(np.stack(one_hot_sequences), dtype=torch.float32)
    expressions_tensor = torch.tensor(expressions, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        recon_sequences, _, _ = model(one_hot_sequences_tensor, expressions_tensor)

    final_sequence = preprocess_for_cnn(one_hot_sequences_tensor, recon_sequences.view(-1, 150, 4))
    predicted_expressions = cnn(final_sequence).cpu().numpy().flatten()

    # Decode the one-hot encoded sequences back to nucleotide sequence
    decoded_sequences = []
    for seq in final_sequence:
        decoded_seq = decode_one_hot(seq.cpu().numpy())
        decoded_sequences.append("".join(decoded_seq))
    
    return decoded_sequences, predicted_expressions

# Set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

def main():

    # Defining hyperparameters
    batch_size = 64
    epochs = 100
    early_stopping_patience = 10
    early_stopping_min_delta = 0.01
    latent_size = 20
    num_masks = 4
    min_mask_len = 5
    max_mask_len = 20

    # Paths to Data and Pre-trained CNN
    path_to_data = '../Data/Combined/LaFleur_supp.csv'
    path_to_cvae = '../Models/CVAE_6_1_2.pt'
    path_to_cnn = '../Models/CNN_6_1_2.keras'
    path_to_summary = '../Testing CVAE/runs/CNN_6_2_summary'

    # Set up device
    device = get_device()

    # Initialize model, optimizer
    cnn = KerasModelWrapper(path_to_cnn)
    model = CVAE(150, latent_size, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Load data, one-hot encode, mask sequences
    onehot_masked, expressions = load_features(path_to_data, num_masks, min_mask_len, max_mask_len)

    # Split data into training and testing sets
    onehot_masked_train, onehot_masked_test, expressions_train, expressions_test = train_test_split(
        onehot_masked, expressions, test_size=0.2, random_state=seed
    )

    # Preprocess sequences and expressions into tensors
    masked_tensor_train = torch.tensor(np.stack(onehot_masked_train), dtype=torch.float32)
    expressions_tensor_train = torch.tensor(expressions_train.values, dtype=torch.float32)
    masked_tensor_test = torch.tensor(np.stack(onehot_masked_test), dtype=torch.float32)
    expressions_tensor_test = torch.tensor(expressions_test.values, dtype=torch.float32)

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(masked_tensor_train, expressions_tensor_train),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(masked_tensor_test, expressions_tensor_test),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )


    # Train and test the model
    train_losses, test_losses = fit_model(epochs,
                                          model,
                                          cnn,
                                          path_to_summary,
                                          train_loader,
                                          test_loader,
                                          optimizer,
                                          device,
                                          early_stopping_patience,
                                          early_stopping_min_delta
    )

    # Plot the training and testing losses
    plot_losses(train_losses, test_losses)

    # Save the model
    save_model(model, path_to_cvae)

    # Load the model
    load_model(path_to_cvae)

    # Test Example
    masked_sequences = ['TTTTCTATCTACGTACTTGACACTATTTCNNNNNNNNNNATTACCTTAGTTTGTACGTT']
    expressions = [0.5]

    # Generate infills
    infilled_sequences, predicted_expressions = generate_infills(model, cnn, masked_sequences, expressions)
    for masked, infilled, expressions in zip(masked_sequences, infilled_sequences, predicted_expressions):
        print("Masked:  ", masked)
        print("Infilled:", infilled)
        print("Predicted Expression:", expressions)
        print()

if __name__ == "__main__":
    main()
