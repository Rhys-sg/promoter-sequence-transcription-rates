import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Lambda # type: ignore
from tensorflow.keras import backend as K # type: ignore

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
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               '0': [0, 0, 0, 0]}  # Placeholder for padding

    return [mapping[nucleotide.upper()] for nucleotide in sequence]

# Custom masking function for output
@tf.keras.utils.register_keras_serializable(package="Custom", name="custom_copy_masked_elements")
def custom_copy_masked_elements(args):
    sequence_input, lstm_output = args
    mask = tf.constant([0, 0, 0, 0, 0], dtype=tf.float32)
    is_masked = tf.reduce_all(tf.equal(sequence_input, mask), axis=-1, keepdims=True)
    # Get the argmax of lstm_output
    output_indices = tf.argmax(lstm_output, axis=-1)
    
    # Create one-hot encoding from indices
    one_hot_output = tf.one_hot(output_indices, depth=4)  # depth=4 for A, T, C, G
    
    # Use one-hot encoded output where not masked
    output = tf.where(is_masked, one_hot_output, sequence_input[..., :4])
    
    return output

def build_lstm_model(sequence_length=150, input_nucleotide_dim=5, output_nucleotide_dim=4, expression_dim=1):
    sequence_input = Input(shape=(sequence_length, input_nucleotide_dim), name='sequence_input')
    expression_input = Input(shape=(sequence_length, expression_dim), name='expression_input')
    combined_input = Concatenate()([sequence_input, expression_input])
    lstm_out = LSTM(128, return_sequences=True)(combined_input)
    lstm_dense_output = Dense(output_nucleotide_dim, activation='softmax')(lstm_out)
    
    # Use the custom function to mask and convert to one-hot
    final_output = Lambda(custom_copy_masked_elements)([sequence_input, lstm_dense_output])
    
    # Define the model
    model = Model(inputs=[sequence_input, expression_input], outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def train_model(lstm_model, cnn_model, X_sequence_train, X_expressions_train, y_train, batch_size, epochs=10, learning_rate=0.01):
    
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
            print('\r')
            print(f'Epoch {epoch + 1}, Sequence {i}/{len(X_sequence_train)}', end='\r')

            # Select batch data
            X_sequence_batch = X_sequence_train[i:i + batch_size]
            X_expressions_batch = X_expressions_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            with tf.GradientTape() as tape:
                predicted_sequence = lstm_model([X_sequence_batch, X_expressions_batch])
                loss = loss_func(predicted_sequence, X_expressions_batch, y_batch, cnn_model)

            # Calculate gradients only for the LSTM model
            gradients = tape.gradient(loss, lstm_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, lstm_model.trainable_variables))
        
        loss_history.append(loss.numpy())

    return loss_history

def loss_func(predicted_sequence, X_expressions_batch, y_train, cnn_model):
    """
    Custom loss function to calculate the MSE between the predicted expression and the true expression.
    It is based on "Expression Consistency Loss," which ensures that the generated DNA sequence has a
    similar expression to the given sequence expression using mean squared error.

    Implementation:
      1) If the predicted sequence is equal to the true sequence, the loss is 0.
      2) Otherwise, the predicted sequence is passed through the CNN model to get the predicted expression.
      3) The loss is then calculated as the MSE between the true expression and the predicted expression.
    
    This function does NOT include
      1) GAN Adversarial Loss: A binary cross-entropy loss used to train the discriminator to differentiate
         between real and generated sequences.
      2) Deviation Loss: A loss that penalizes the deviation of the generated sequence from the unmasked
         sections of the input sequence.

    """
    if predicted_sequence == y_train:
        return 0
    predicted_expression = cnn_model(predicted_sequence)
    return tf.reduce_mean(tf.square(X_expressions_batch - predicted_expression))


def evaluate_model(lstm_model, cnn_model, X_sequence_test, X_expressions_test):
    X_expressions_test = np.expand_dims(X_expressions_test, axis=-1)
    X_expressions_test = np.repeat(X_expressions_test, X_sequence_test.shape[1], axis=1)

    print(X_sequence_test.shape)
    print(X_expressions_test.shape)

    predicted_sequence = lstm_model.predict([X_sequence_test, X_expressions_test])
    predicted_expression = cnn_model.predict(predicted_sequence)
    mse = tf.reduce_mean(tf.square(X_expressions_test - predicted_expression)).numpy()
    
    return mse, predicted_expression

def predict_with_lstm(lstm_model, sequence, expression, scaler, max_length=150):
    one_hot_encoded_sequence = np.array([one_hot_encode_input(apply_padding(sequence, max_length))])
    normalized_expression = scaler.transform([[expression]])
    normalized_expression = np.repeat(normalized_expression, max_length, axis=1)
    predicted_sequence = lstm_model.predict([one_hot_encoded_sequence, normalized_expression])

    print(predicted_sequence[0])
    
    return onehot_decode_output(predicted_sequence)

def onehot_decode_output(sequence):
    mapping = {(1, 0, 0, 0): 'A',
               (0, 1, 0, 0): 'T',
               (0, 0, 1, 0): 'C',
               (0, 0, 0, 1): 'G',
               (0, 0, 0, 0): '0'}  # Placeholder for padding
    
    # Ensure the output is 2D and we iterate over the actual one-hot encoded nucleotides
    sequence = np.squeeze(sequence)    
    return ''.join([mapping[tuple(round(n) for n in nucleotide)] for nucleotide in sequence])


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
