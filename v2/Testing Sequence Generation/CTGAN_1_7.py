import pandas as pd

def expand_sequences(df, max_length=150):
    expanded_data = []

    for _, row in df.iterrows():
        sequence = row['Promoter Sequence']
        expression = row['Normalized Expression']
        padded_sequence = sequence.ljust(max_length, '-')
        sequence_dict = {f'Base_{i}': padded_sequence[i] for i in range(max_length)}
        sequence_dict['Normalized Expression'] = expression
        expanded_data.append(sequence_dict)

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df
