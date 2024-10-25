import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os

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
    encoded = np.array([encoding[base.upper()] for base in padded_seq], dtype=np.float32)
    return encoded

def mask_sequence(sequence, max_mask_length=20):
    seq_length = len(sequence)
    mask_length = random.randint(5, max_mask_length)
    start = random.randint(0, seq_length - mask_length)

    # Replace the masked section with 'N'
    masked_seq = sequence[:start] + 'N' * mask_length + sequence[start + mask_length:]
    return masked_seq

def load_and_preprocess_data(train_path, test_path, seq_length=150):
    # Load datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Apply masking and one-hot encoding
    X_train = np.stack(
        train_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length))
    )
    X_test = np.stack(
        test_data['Promoter Sequence'].apply(lambda x: one_hot_encode_sequence(mask_sequence(x), seq_length))
    )

    # Extract normalized expression levels
    y_train = train_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)
    y_test = test_data['Normalized Expression'].values.reshape(-1, 1).astype(np.float32)

    return (X_train, y_train), (X_test, y_test)

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(150, 4)),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim + 1,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(150 * 4, activation='sigmoid'),
            tf.keras.layers.Reshape((150, 4))
        ])

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, x, condition):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        z_cond = tf.concat([z, condition], axis=1)
        return self.decoder(z_cond), mean, logvar

def compute_loss(model, x, condition):
    reconstruction, mean, logvar = model(x, condition)
    reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction))
    kl_loss = -0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)
    return reconstruction_loss + kl_loss

@tf.function
def train_step(model, x, condition, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, condition)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(model, optimizer, train_dataset, epochs=10):
    for epoch in range(epochs):
        for train_x, train_cond in train_dataset:
            loss = train_step(model, train_x, train_cond, optimizer)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy():.4f}')

# Evaluation function
def evaluate_model(model, test_dataset):
    losses = []
    for test_x, test_cond in test_dataset:
        loss = compute_loss(model, test_x, test_cond)
        losses.append(loss.numpy())
    return np.mean(losses)

def main():
    # Load and preprocess data
    train_path = 'v2/Data/Train Test/train_data.csv'
    test_path = 'v2/Data/Train Test/test_data.csv'
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data(train_path, test_path)

    # Initialize model and optimizer
    latent_dim = 16
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Prepare datasets
    batch_size = 32
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

    # Train the model
    train_model(model, optimizer, train_dataset, epochs=10)

    # Evaluate the model
    test_loss = evaluate_model(model, test_dataset)
    print(f'Average Test Loss: {test_loss:.4f}')

# Run the main function
if __name__ == '__main__':
    main()
