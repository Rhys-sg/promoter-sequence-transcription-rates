import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DatabaseQuery:
    def __init__(self, file_path='../Data/combined/LaFleur_supp.csv', max_length=150):
        self.df = pd.read_csv(file_path)[['File Name', 'Observed log(TX/Txref)', 'Promoter Sequence']]
        self.df['Normalized Observed log(TX/Txref)'] = MinMaxScaler().fit_transform(self.df[['Observed log(TX/Txref)']])
        self.df['Length'] = self.df['Promoter Sequence'].apply(lambda x: len(x))
        self.max_length = max_length
    
    def apply_padding(self, sequence,):
        return '0' * (self.max_length - len(sequence)) + sequence
    
    def query_sequences(self, sequence_1, expression_1, max_bp_difference=0):
        matched_sequences = []
        length = len(sequence_1)
        sub_df = self.df[self.df['Length'] == length]
        for index, row in sub_df.iterrows():
            sequence_2 = row['Promoter Sequence']
            expression_2 = row['Normalized Observed log(TX/Txref)']
            expression_difference = expression_1 - expression_2
            abs_expression_difference = abs(expression_difference)
            bp_difference = 0

            for i in range(length):
                if sequence_1[i] != '_' and sequence_1[i] != sequence_2[i]:
                    bp_difference += 1

            row_data = {'Sequence' : sequence_2,
                    'BP Distance' : bp_difference,
                    'Expression' : expression_2,
                    'Expression Difference' : expression_difference,
                    'Abs Expression Difference' : abs_expression_difference
            }
            matched_sequences.append(row_data)
        
        matched_sequences_df = pd.DataFrame(matched_sequences)
        return matched_sequences_df[matched_sequences_df['BP Distance'] <= max_bp_difference].sort_values('Absolute Expression Difference')