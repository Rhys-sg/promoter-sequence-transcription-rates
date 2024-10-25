import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import layers, Model
import random

# File paths
TRAIN_DATA_PATH = 'v2/Data/Train Test/train_data.csv'
TEST_DATA_PATH = 'v2/Data/Train Test/test_data.csv'

# Load and preprocess data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Extract sequences and expression values
    train_sequences = train_data['Promoter Sequence'].values
    train_expressions = train_data['Normalized Expression'].values
    test_sequences = test_data['Promoter Sequence'].values
    test_expressions = test_data['Normalized Expression'].values

    return train_sequences, train_expressions, test_sequences, test_expressions

# Mask a continuous section in each sequence
def mask_sequence(sequence, min_len=5, max_len=20):
    seq_len = len(sequence)
    mask_len = random.randint(min_len, max_len)
    start = random.randint(0, seq_len - mask_len)
    masked_seq = sequence[:start] + 'N' * mask_len + sequence[start + mask_len:]
    return masked_seq

# One-hot encode a sequence
def one_hot_encode_sequence(seq, length=150):
    encoding = {'A': [1, 0, 0, 0],
                'T': [0, 1, 0, 0],
                'C': [0, 0, 1, 0], 
                'G': [0, 0, 0, 1],
                'N': [0.25, 0.25, 0.25, 0.25],
                '0': [0, 0, 0, 0]}
    padded_seq = seq.ljust(length, '0')
    return torch.tensor([encoding[base.upper()] for base in padded_seq], dtype=torch.float32)

# Define the CVAE model
class CVAE(Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(150, 4)),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim)  # For mean and log variance
        ])

        # Decoder
        self.decoder_net = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim + 1,)),  # Latent vector + expression level
            layers.Dense(75 * 64, activation='relu'),
            layers.Reshape((75, 64)),
            layers.Conv1DTranspose(64, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(32, 3, activation='relu', padding='same'),
            layers.Conv1DTranspose(4, 3, activation='sigmoid', padding='same')
        ])

    def encode(self, x):
        mean_logvar = self.encoder_net(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, expression):
        z = tf.concat([z, tf.expand_dims(expression, 1)], axis=1)
        return self.decoder_net(z)

    def call(self, x, expression):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, expression)

# Loss function for CVAE
def cvae_loss(x, x_recon):
    return tf.reduce_mean(tf.square(x - x_recon))

# Train the CVAE
def train_cvae(model, train_sequences, train_expressions, epochs=10, batch_size=32):
    optimizer = tf.keras.optimizers.Adam()

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_sequences, train_expressions)
    ).shuffle(10000).batch(batch_size)

    for epoch in range(epochs):
        for batch_sequences, batch_expressions in train_dataset:
            with tf.GradientTape() as tape:
                recon_sequences = model(batch_sequences, batch_expressions)
                loss = cvae_loss(batch_sequences, recon_sequences)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# Main function to run everything
def main():
    # Load and preprocess the data
    train_sequences, train_expressions, test_sequences, test_expressions = load_data(
        TRAIN_DATA_PATH, TEST_DATA_PATH
    )

    # Mask sequences in training and testing data
    train_sequences = [mask_sequence(seq) for seq in train_sequences]
    test_sequences = [mask_sequence(seq) for seq in test_sequences]

    # One-hot encode the sequences
    train_sequences = torch.stack([one_hot_encode_sequence(seq) for seq in train_sequences])
    test_sequences = torch.stack([one_hot_encode_sequence(seq) for seq in test_sequences])

    # Initialize the CVAE model
    latent_dim = 16
    cvae = CVAE(latent_dim)

    # Train the CVAE
    train_cvae(cvae, train_sequences.numpy(), train_expressions)

    # Evaluate the model on test data
    test_recon = cvae(test_sequences.numpy(), test_expressions)
    loss = cvae_loss(test_sequences.numpy(), test_recon)
    print(f'Test Loss: {loss.numpy()}')

if __name__ == '__main__':
    main()
