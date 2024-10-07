import torch
import torch.nn as nn

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Encoder LSTM
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Decoder LSTM
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)

        # Final output layer (to convert LSTM outputs to desired output size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, target_length):
        # Initialize hidden and cell state
        batch_size = input_seq.size(0)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(input_seq.device)

        # Encoder
        encoder_outputs, (hidden, cell) = self.encoder(input_seq, (hidden, cell))

        # Initialize the first decoder input (can be zeros or the last encoder output)
        decoder_input = torch.zeros(batch_size, 1, encoder_outputs.size(2)).to(input_seq.device)
        decoder_outputs = []

        # Decoder loop (use the target_length to control output sequence length)
        for t in range(target_length):
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.fc(decoder_output)
            decoder_outputs.append(output)
            decoder_input = output  # Feed the output as the next input

        # Concatenate the outputs to form the final sequence
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

# Define the model
input_size = 10  # Number of features in the input sequence
hidden_size = 50  # Hidden state size of the LSTM
output_size = 10  # Number of features in the output sequence
target_length = 20  # Desired output sequence length

model = Seq2SeqLSTM(input_size, hidden_size, output_size)

# Dummy input sequence (batch_size=2, seq_len=15, input_size=10)
input_seq = torch.randn(2, 15, input_size)

# Forward pass
output_seq = model(input_seq, target_length)

print(output_seq.shape)  # Output: (2, 20, 10) - (batch_size, target_length, output_size)
