import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample data
# Let's assume each input sequence has a maximum length of 10
max_input_length = 10
max_output_length = 5
num_samples = 1000

# Generate random input sequences (batch_size, max_input_length, num_features)
X = np.random.rand(num_samples, max_input_length, 1)

# Generate output sequences with variable lengths
# For simplicity, we'll generate outputs that have a length of 1 to max_output_length
Y = []
output_lengths = np.random.randint(1, max_output_length + 1, num_samples)  # Random output lengths

for i in range(num_samples):
    # Create an output sequence of random values of the specified length
    Y.append(np.random.rand(output_lengths[i], 1))

# Pad the output sequences to have the same shape
Y_padded = tf.keras.preprocessing.sequence.pad_sequences(Y, padding='post', maxlen=max_output_length, dtype='float32')

# Convert output lengths to a NumPy array
output_lengths = np.array(output_lengths)

# Create the model
encoder_input = layers.Input(shape=(max_input_length, 1))
encoder_lstm = layers.LSTM(64, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_input)
encoder_states = [state_h, state_c]

# Decoder setup
decoder_input = layers.Input(shape=(None, 1))  # None for dynamic length
decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)

# Output layer
decoder_dense = layers.Dense(1, activation='linear')  # Change to appropriate activation as needed
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = keras.Model([encoder_input, decoder_input], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Prepare decoder input data (teacher forcing)
decoder_input_data = np.zeros((num_samples, max_output_length, 1))  # For teacher forcing

# Train the model
model.fit([X, decoder_input_data], Y_padded, epochs=10, batch_size=32)

# To generate predictions (example)
# You need to input the encoder sequence and desired output length
def generate_sequence(input_seq, output_length):
    # Encode the input sequence
    encoder_output, state_h, state_c = model.layers[1](input_seq[np.newaxis, :, :])  # Encoder LSTM
    states = [state_h, state_c]  # Get the hidden and cell states

    # Prepare the initial input for the decoder
    decoder_input = np.zeros((1, max_output_length, 1))  # Start with zeros
    predictions = []

    for t in range(output_length):
        output_seq, state_h, state_c = model.layers[2](decoder_input, initial_state=states)  # Decoder LSTM
        states = [state_h, state_c]  # Update states
        output = model.layers[3](output_seq)  # Dense layer
        predictions.append(output.numpy())

        # Update the decoder input (teacher forcing is not used here)
        decoder_input[0, t, :] = output.numpy()  # Use predicted output

    return np.array(predictions).squeeze()

# Example input sequence and output length
example_input = np.random.rand(max_input_length, 1)
predicted_output = generate_sequence(example_input, output_length=3)

print(predicted_output)
