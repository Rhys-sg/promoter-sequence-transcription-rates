import random
import pandas as pd
import numpy as np
import torch

def mask_predict_data(df, cnn, num_masks=1, num_inserts=1, min_mask=1, max_mask=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = get_device()

    masked_df = mask_data(df, num_masks, num_inserts, min_mask, max_mask)
    return preprocess_data(masked_df, cnn, device)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mask_data(df, num_masks, num_inserts, min_mask, max_mask):
        
    original_sequences = df['Promoter Sequence']
    masked_sequences = []
    infilled_sequences = []
    mask_lengths = []
    mask_starts = []

    for sequence in original_sequences:
        for _ in range(num_masks):
            mask_length = random.randint(min_mask, max_mask)
            mask_start = random.randint(0, len(sequence) - mask_length)

            # Mask the sequence
            masked_seq = sequence[:mask_start] + 'N' * mask_length + sequence[mask_start + mask_length:]

            # Add the original, unmasked sequence
            masked_sequences.append(masked_seq)
            infilled_sequences.append(sequence)
            mask_lengths.append(mask_length)
            mask_starts.append(mask_start)

            # Generate multiple random infills for this masked sequence
            for _ in range(num_inserts):
                random_infill = ''.join(random.choices('ATCG', k=mask_length))
                random_infilled_seq = masked_seq[:mask_start] + random_infill + masked_seq[mask_start + len(random_infill):]

                # Collect results
                masked_sequences.append(masked_seq)
                infilled_sequences.append(random_infilled_seq)
                mask_lengths.append(mask_length)
                mask_starts.append(mask_start)

    # Construct the new DataFrame with required columns
    new_df = pd.DataFrame({
        'Masked Promoter Sequence': masked_sequences,
        'Infilled Promoter Sequence': infilled_sequences,
        'Mask Length': mask_lengths,
        'Mask Start': mask_starts,
    })

    return new_df

def preprocess_data(df, cnn, device):

    def one_hot_sequence(seq, length=150):
        seq = seq.ljust(length, '0')
        mapping = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1],
                   '0': [0, 0, 0, 0]} # Padding
        return np.array([mapping[nucleotide.upper()] for nucleotide in seq])

    # Apply one-hot encoding and batch the sequences
    augmented_sequences = df['Infilled Promoter Sequence'].apply(lambda x: one_hot_sequence(x))
    augmented_sequences_tensor = torch.tensor(np.stack(augmented_sequences), dtype=torch.float32).to(device)

    # Predict expressions using the CNN in batches (disable gradients)
    with torch.no_grad():
        expressions = cnn(augmented_sequences_tensor)

    df['Expressions'] = expressions

    return df