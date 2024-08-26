import pandas as pd
import numpy as np

def preprocess_data(df):

    UP_swap_dict = {
        'gourse-136fold-up': 'GAAAATATATTTTTCAAAAGTA',
        'gourse-326fold-up': 'GGAAAATTTTTTTTCAAAAGTA',
        'noUP': '',
    }
    Minus35_swap_dict = {
        '35T->A/33G->T': 'ATTACA', 
        '34G->T/30A->C': 'TTTACC', # '33G->T/30A->C'
        '34G->T': 'TTTACA', # '33G->T'
        '35T->C/33G->T/31G->C': 'CTTAGA', # '35T->C/33G->T/31C->G'
        'consensus35': 'TTGACA',
        '32A->C/31C->A': 'TTGCAA',
        '35T->C/33G->C/31C->G': 'CTCAGA',
        '33G->A/31C->G': 'TTAAGA',
    }
    Spacer_swap_dict = {
        'ECK125137405': 'AAAACTCATTTTATTTT',
        'ECK125137108': 'AGCACGAAAATGGAAGT',
        'ECK125136938': 'ATAACTTAGAAAGTAAT',
        'ECK125137726': 'TTTCCATTAGCGAGTAT',
        'ECK125137104': 'TCGCGCATGATCGAAAG',
        'ECK125137640': 'TGGCTGAATGGTCTGTC',
        'lac-spacer-17bp': 'CTTTATGCTTCGGCTCG',
        'P1-6-17bp': 'CTTTATGCTTTTATGTT'
    }
    Minus10_swap_dict = {
        '7T->A': 'TATAAA',
        '12T->G/7T->C': 'GATAAC',
        '12T->G/11A->T/9A->G/8A->T/7T->A': 'GTTGTA',
        'consensus10': 'TATAAT',
        '9A->G/8A->T': 'TATGTT',
        '12T->G': 'GATAAT',
        '12T->G/11A->T/7T->A': 'GTTAAA',
        '12T->A': 'AATAAT',
    }

    df['UP_element'] = df['UP_element'].replace(UP_swap_dict)
    df['Minus35'] = df['Minus35'].replace(Minus35_swap_dict)
    df['Spacer'] = df['Spacer'].replace(Spacer_swap_dict)
    df['Minus10'] = df['Minus10'].replace(Minus10_swap_dict)

  
    for i, row in df.iterrows():
        df.at[i, 'sub_variant'] = get_substrings(row['variant'], row['Minus35'], row['Minus10'])
        df.at[i, 'log_transformed'] = np.log(row['RNA_exp_average'])
        
    return df

def get_substrings(sequence, minus35, minus10):

    index_str2 = sequence.find(minus35)
    index_str3 = sequence.find(minus10)

    start_index = max(0, index_str2 - 16)
    end_index = min(len(sequence), index_str3 + len(minus10) + 21)

    return sequence[start_index:end_index]


if __name__ == '__main__':
    # df = pd.read_csv('v2/Data/rlp5Min_SplitVariants.txt', delimiter=' ')[['variant', 'UP_element', 'Minus35', 'Spacer', 'Minus10', 'RNA_exp_average']]
    df = pd.read_csv('v2/Data/rlp5Min_SplitVariants.txt', delimiter=' ')
    df.to_csv('v2/Data/rlp5Min_SplitVariants.csv', index=False)
    # df = preprocess_data(df)
    # df.to_csv('v2/Data/rlp5Min_SplitVariants_Processed.csv', index=False)
