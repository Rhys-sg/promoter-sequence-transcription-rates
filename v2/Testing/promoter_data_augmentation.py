import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def promoter_data_augmentation():
    LaFleur_df = pd.read_csv('v2/data/41467_2022_32829_MOESM5_ESM.csv')
    ProD_df = pd.read_csv('v2/data/ProD_sigma70_spacer_data.csv')

    LaFleur_df = LaFleur_df[['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR', 'Observed log(TX/Txref)']]
    LaFleur_df.columns = ['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR', 'Observed']
    ProD_df = ProD_df[['spacer', 'assumed_observed']]

    LaFleur_df['Observed'] = MinMaxScaler().fit_transform(LaFleur_df[['Observed']])
    ProD_df['assumed_observed'] = MinMaxScaler().fit_transform(ProD_df[['assumed_observed']])


    augmented_data = {
        'UP': [],
        'h35': [],
        'spacs': [],
        'h10': [],
        'disc': [],
        'ITR': [],
        'Observed': [],
    }

    # Set a random seed for reproducibility
    random_seed = 42

    """
    According to LaFleur, 16% of their model variance could be explained by the spacer sequence.
    Looking at their model (see v2/Comparisons/LaFleur_promoter.ipynb), variations in the spacer have:
    - a mean range of 0.303
    - a median range of 0.330
    - a mode range of 0.022
    - a standard deviation of 0.252

    When looking at variation of spacers with the same length (all ProD sigma 70 spacers are 17 bp), we see:
    - a mean range of 0.282
    - a median range of 0.206
    - a mode range of 0.022
    - a standard deviation of 0.232

    This data augmentation uses the median (0.206) because it is is less affected by extreme values or outliers compared to the mean.

    """
    spacer_variance = 0.206


    for i, row in LaFleur_df.iterrows():
        if i % 100 == 0:
            print(f'{i}/{len(LaFleur_df)} rows processed', end='\r')

        # ProD data does not vary spacer length, their sigma70 has 17 nucleotides
        if len(row['spacs']) != 17:
            continue
        try:
            # Swap out the spacer for a random one from ProD data, adjust the observed value
            for i in np.arange(0, 1.1, 0.1):

                filtered_df = ProD_df[ProD_df['assumed_observed'] == i]
                
                ProD_spacer, ProD_observed = filtered_df.sample(random_state=random_seed).values[0]

                adjusted_transcription = row['Observed'] * (1 + (ProD_observed * spacer_variance) - (spacer_variance / 2))
                
                augmented_data['UP'].append(row['UP'])
                augmented_data['h35'].append(row['h35'])
                augmented_data['spacs'].append(ProD_spacer)
                augmented_data['h10'].append(row['h10'])
                augmented_data['disc'].append(row['disc'])
                augmented_data['ITR'].append(row['ITR'])
                augmented_data['Observed'].append(adjusted_transcription)
        except:
            print('an error occurred with row:', row)

    augmented_df = pd.DataFrame(augmented_data)

    augmented_df = pd.concat([LaFleur_df, augmented_df])

    augmented_df['Observed'] = MinMaxScaler().fit_transform(augmented_df[['Observed']])
    
    augmented_df.to_csv('v2/data/augmented_data_1.0.csv', index=False)

    print(augmented_df.head())
    print(augmented_df.describe())


if __name__ == '__main__':
    promoter_data_augmentation()