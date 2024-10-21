import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Lambda # type: ignore
import keras

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    scaler = MinMaxScaler()
    df['Normalized Observed log(TX/Txref)'] = scaler.fit_transform(df[['Observed log(TX/Txref)']])
    return df, scaler

def combine_columns(df):
    X = df[['Promoter Sequence']].astype(str).agg(''.join, axis=1)
    y = df['Normalized Observed log(TX/Txref)']
    return X, y

def preprocess_X_y(df, num_augmentations=1):
    sequences, expressions = combine_columns(df)

    X_sequence = []
    X_expressions = []
    y = []

    for full_sequence, expression in zip(sequences, expressions):
        for _ in range(num_augmentations):
            len_removed = random.randint(1, 10)
            masked_sequence, missing_element = remove_section_get_features(full_sequence, len_removed)

            X_sequence.append(one_hot_encode_input(apply_padding(masked_sequence, 150)))
            X_expressions.append(expression)
            y.append(one_hot_encode_output(apply_padding(full_sequence, 150)))

    return np.array(X_sequence), np.array(X_expressions), np.array(y)

def remove_section_get_features(sequence, section_length):
    seq_length = len(sequence)
    start_idx = random.randint(0, seq_length - section_length)
    missing_element = sequence[start_idx:start_idx + section_length]
    masked_sequence = sequence[:start_idx] + '_' * section_length + sequence[start_idx + section_length:]
    return masked_sequence, missing_element

def apply_padding(sequence, max_length):
    return '0' * (max_length - len(sequence)) + sequence

def one_hot_encode_input(sequence):
    mapping = {'A': [1, 0, 0, 0, 0],
               'T': [0, 1, 0, 0, 0],
               'C': [0, 0, 1, 0, 0],
               'G': [0, 0, 0, 1, 0],
               '0': [0, 0, 0, 0, 1],  # Placeholder for masking
               '_': [0, 0, 0, 0, 0]}  # Placeholder for padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

def one_hot_encode_output(sequence):
    mapping = {'A': [1, 0, 0, 0, 0],
               'T': [0, 1, 0, 0, 0],
               'C': [0, 0, 1, 0, 0],
               'G': [0, 0, 0, 1, 0],
               '0': [0, 0, 0, 0, 1]}  # Placeholder for padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

# Register the custom function
@keras.saving.register_keras_serializable(package="Custom", name="custom_copy_masked_elements")
def custom_copy_masked_elements(args):
    sequence_input, lstm_output = args
    mask = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32)
    is_masked = tf.reduce_all(tf.equal(sequence_input, mask), axis=-1, keepdims=True)
    output = tf.where(is_masked, lstm_output, sequence_input[..., :5])
    
    return output

def build_lstm_model(sequence_length=150, input_nucleotide_dim=5, output_nucleotide_dim=5, expression_dim=1):
    # Input layers
    sequence_input = Input(shape=(sequence_length, input_nucleotide_dim), name='sequence_input')
    expression_input = Input(shape=(sequence_length, expression_dim), name='expression_input')

    # Combine inputs
    combined_input = Concatenate()([sequence_input, expression_input])
    lstm_out = LSTM(128, return_sequences=True)(combined_input)
    lstm_dense_output = Dense(output_nucleotide_dim, activation='softmax')(lstm_out)

    # Masked output based on custom logic
    masked_output = Lambda(custom_copy_masked_elements)([sequence_input, lstm_dense_output])

    # Model for training (using softmax output)
    model = Model(inputs=[sequence_input, expression_input], outputs=masked_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(lstm_model, cnn_model, X_sequence_train, X_expressions_train, y_train, batch_size=512, epochs=1, learning_rate=0.01):
    
    # Freeze CNN model layers and Ensure LSTM model layers are trainable
    for layer in cnn_model.layers:
        layer.trainable = False

    for layer in lstm_model.layers:
        layer.trainable = True

    X_expressions_train = np.expand_dims(X_expressions_train, axis=-1)
    X_expressions_train = np.repeat(X_expressions_train, X_sequence_train.shape[1], axis=1)
    y_train = np.repeat(y_train, X_sequence_train.shape[1], axis=1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data at the beginning of each epoch
        indices = np.arange(X_sequence_train.shape[0])
        np.random.shuffle(indices)
        
        # Shuffle both input datasets and output datasets, maintaining order
        X_sequence_train = X_sequence_train[indices]
        X_expressions_train = X_expressions_train[indices]
        y_train = y_train[indices]

        # Process data in batches
        for i in range(0, len(X_sequence_train), batch_size):
            print(f'Epoch {epoch + 1}, Sequence {i}/{len(X_sequence_train)}')

            # Select batch data
            X_sequence_batch = X_sequence_train[i:i + batch_size]
            X_expressions_batch = X_expressions_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                predicted_sequence = lstm_model([X_sequence_batch, X_expressions_batch])
                loss_total = loss_func(predicted_sequence, X_expressions_batch, y_batch, cnn_model)

            # Calculate gradients only for the LSTM model
            gradients = tape.gradient(loss_total, lstm_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, lstm_model.trainable_variables))
        
        loss_history.append(loss_total.numpy())

    return loss_history

def loss_func(predicted_sequences_batch, X_expressions_batch, y_train, cnn_model, weight_one_hot=0.000005, weight_expression=30):

    # Get one-hot encoded sequences
    one_hot_batch_len_5, one_hot_batch_len_4 = get_argmax_STE_one_hot(predicted_sequences_batch)

    # Calculate each loss component individually, apply weights
    weighted_one_hot_loss = weight_one_hot * tf.reduce_mean(loss_func_one_hot_deviation(predicted_sequences_batch, one_hot_batch_len_5))
    weighted_expression_loss = weight_expression * tf.reduce_mean(loss_func_cnn(cnn_model, one_hot_batch_len_4, X_expressions_batch))
    
    print(f'Weighted One-Hot Loss: {weighted_one_hot_loss.numpy()}')
    print(f'Weighted Expression Loss: {weighted_expression_loss.numpy()}')

    return weighted_one_hot_loss + weighted_expression_loss

def loss_func_one_hot_deviation(predicted_sequences_batch, one_hot_batch):
    """
    This uses Categorical Crossentropy Loss to calculate the distance between the predicted 
    and the one-hot encoded predicted sequences.

    Alternatively, we can use:
    1. Kullback-Leibler Divergence (KL Divergence) Loss
    2. Binary Crossentropy Loss
    3. Mean Squared Error (MSE) Loss

    """
    return tf.keras.losses.categorical_crossentropy(predicted_sequences_batch, one_hot_batch)

def loss_func_cnn(cnn_model, one_hot_batch_len_4, X_expressions_batch):
    """
    Predict the expression from the LSTM-predicted sequence and calculate the mean squared error

    """
    predicted_expression = cnn_model(one_hot_batch_len_4)
    return tf.reduce_mean(tf.square(X_expressions_batch - predicted_expression))

def get_argmax_STE_one_hot(predicted_sequences_batch):
    """
    Returns the one-hot encoded version of the argmax of the softmax output:

    1. Perform the softmax to get probabilities, 
    2. Use argmax to get the index of the maximum value
    3. Create a one-hot encoded version of the argmax indices
    4. Directly return the one-hot output

    Because the CNN requires a sequence of length 4, we remove the last element of the one-hot encoded sequence
    It is redundant, but usefull argmax in the LSTM.

    """
    one_hot_batch_len_5 = tf.one_hot(tf.argmax(tf.nn.softmax(predicted_sequences_batch), axis=-1), depth=tf.shape(predicted_sequences_batch)[-1])
    one_hot_batch_len_4 = np.delete(one_hot_batch_len_5, -1, axis=-1)

    return one_hot_batch_len_5, one_hot_batch_len_4


def evaluate_model(lstm_model, cnn_model, X_sequence_test, X_expressions_test):
    X_expressions_test = np.expand_dims(X_expressions_test, axis=-1)
    X_expressions_test = np.repeat(X_expressions_test, X_sequence_test.shape[1], axis=1)
    predicted_sequence = lstm_model.predict([X_sequence_test, X_expressions_test])
    one_hot_batch_len_5, one_hot_batch_len_4 = get_argmax_STE_one_hot(predicted_sequence)
    predicted_expression = cnn_model.predict(one_hot_batch_len_4)
    mse = tf.reduce_mean(tf.square(X_expressions_test - predicted_expression)).numpy()
    
    return mse, predicted_expression

def predict_with_lstm(lstm_model, sequence, expression, scaler, max_length=150, decode_output=True):
    one_hot_encoded_sequence = np.array([one_hot_encode_input(apply_padding(sequence, max_length))])
    normalized_expression = scaler.transform([[expression]])
    normalized_expression = np.repeat(normalized_expression, max_length, axis=1)
    predicted_masked_sequence = lstm_model.predict([one_hot_encoded_sequence, normalized_expression])

    if not decode_output:
        return predicted_masked_sequence
    
    return one_hot_decode_output(predicted_masked_sequence)

def one_hot_decode_output(sequence):
    mapping = {(1, 0, 0, 0): 'A',
               (0, 1, 0, 0): 'T',
               (0, 0, 1, 0): 'C',
               (0, 0, 0, 1): 'G',
               (0, 0, 0, 0): ''}  # Placeholder for padding
    
    one_hot_batch_len_5, one_hot_batch_len_4 = get_argmax_STE_one_hot(sequence)
    
    # Ensure the output is 2D and we iterate over the actual one-hot encoded nucleotides
    return ''.join([mapping[tuple(nucleotide)] for nucleotide in np.squeeze(one_hot_batch_len_4)])


if __name__ == '__main__':
        
    print('Loading and preprocessing data...')
    file_path = 'v2/Data/combined/LaFleur_supp.csv'
    df, scaler = load_and_preprocess_data(file_path)

    print('Preparing training and test data...')
    X_sequence, X_expressions, y = preprocess_X_y(df)

    print(X_sequence.shape, X_expressions.shape, y.shape)
    X_sequence_train, X_sequence_test, X_expressions_train, X_expressions_test, y_train, y_test = train_test_split(
        X_sequence, X_expressions, y, test_size=0.2, random_state=42)

    print('Building/loading models...')
    cnn_model = load_model('v2/Models/CNN_5_0.keras')
    lstm_model = build_lstm_model(sequence_length=150, input_nucleotide_dim=5, output_nucleotide_dim=4, expression_dim=1)

    print('Training the models...')
    loss_history = train_model(lstm_model, cnn_model, X_sequence_train, X_expressions_train, y_train, batch_size=512, epochs=10, learning_rate=0.01)

    print('Evaluating the models...')
    mse, predicted_expression = evaluate_model(lstm_model, cnn_model, X_sequence_test, X_expressions_test)
    print(f'Mean Squared Error on Test Data: {mse:.4f}')
