import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def similarity(seq1, seq2):
    min_len = min(len(seq1), len(seq2))
    max_len = max(len(seq1), len(seq2))
    return (sum([1 for i in range(min_len) if seq1[i] == seq2[i]]) / len(seq1)) + max_len - min_len

def graph_bp_exp_similarity(similarity_df, sample_size=None):
    if sample_size:
        similarity_df = similarity_df.sample(n=sample_size)
    
    plt.scatter(similarity_df['similarity'], similarity_df['expression_difference'])
    plt.xlabel('Similarity')
    plt.ylabel('Expression Difference')
    plt.title('Similarity vs. Expression Difference')
    plt.show()

def compare(similarity_data):
    for i, row1 in LaFleur.iterrows():
        if i % 10 == 0:
            print(f'{round((i / len(LaFleur)) * 100, 4)} %', end='\r')
        for j, row2 in Urtecho.iterrows():
            similarity_data['LaFleur_seq'].append(row1['sequence'])
            similarity_data['Urtecho_seq'].append(row2['sub_variant'])
            similarity_data['similarity'].append(similarity(row1['sequence'], row2['sub_variant']))
            similarity_data['dG'].append(row1['Observed'])
            similarity_data['norm_RNA_expression'].append(row2['norm_RNA_expression'])
            similarity_data['expression_difference'].append(abs(row1['Observed'] - row2['norm_RNA_expression']))
    return similarity_data
    


if __name__ == '__main__':
    Urtecho = pd.read_csv('v2/Data/rlp5Min_SplitVariants_Processed.csv')
    Urtecho['norm_RNA_expression'] = MinMaxScaler().fit_transform(Urtecho['log_transformed'].values.reshape(-1, 1))
    LaFleur = pd.read_csv('v2/Data/41467_2022_32829_MOESM5_ESM.csv')
    LaFleur['sequence'] = LaFleur[['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR']].astype(str).agg(''.join, axis=1)
    LaFleur['norm_observed'] = MinMaxScaler().fit_transform(LaFleur['Observed'].values.reshape(-1, 1))

    similarity_data = {
        'LaFleur_seq': [],
        'Urtecho_seq': [],
        'similarity': [],
        'dG': [],
        'norm_RNA_expression': [],
        'expression_difference': []
    }

    running = True
    try:
        while running:
            similarity_data = compare(similarity_data)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        similarity_df = pd.DataFrame(similarity_data)
        similarity_df.to_csv('v2/Comparisons/Urtecho/Urtecho_LaFleur_overlap.csv', index=False)

    

        similarity_df = pd.read_csv('v2/Data/Urtecho_LaFleur_overlap.csv')

        graph_bp_exp_similarity(similarity_df, 1000)


