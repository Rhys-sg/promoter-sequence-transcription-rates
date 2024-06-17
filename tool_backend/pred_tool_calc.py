from keras.saving import load_model
import numpy as np
from itertools import combinations_with_replacement
from sortedcontainers import SortedDict
import sys
import os
from functools import reduce

class SuppressOutput:
    def __enter__(self):
        self.stdout_original = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.stdout_original

def pred_trans(seq, model_path):
    """
    Tool to predict the transcription rate of a novel sequence

    Parameters:
    - seq (str[]): UP, h35, spacs, h10, disc, ITR promoter sequences
    - model_path (str): path to the model

    Returns:
    - Predicted transcription rate (float)

    """
    if len(seq) != 6:
        return "The input sequence must have 6 elements."
    try:
        enc_seq = one_hot(seq)
        print(enc_seq)
        print(enc_seq[0].shape)
        model = load_model(model_path)
        with SuppressOutput():
            rate = model.predict(enc_seq)
    except Exception as e:
        return f"An exception occurred: {e}"
    return rate[0][0]


def pred_prom(model_path, target, tolerance=float('inf'), max_results=None, max_iter=None,
              UP=None, h35=None, spacs=None, h10=None, disc=None, ITR=None):
    """
    Tool to predict the promoter sequences closest to the target

    This applies a combinatorial approach. It simulates all possible combinations of
    promoter sequences (not given), encodes them, and predicts the transcription rate.
    The closest predictions to the target are returned. Alternative approaches include
    an inverse transformation, specialized model, and database. These are more efficient.
    However, they are not exhaustive models and will exclude sequences.

    Parameters:
    - model_path (str): path to the model
    - target (float): target value to predict
    - tolerance (float): maximum difference between the predicted value and the target
    - results (int): number of results to return
    - max_iter (int): maximum number of iterations to run
    - UP, h35, spacs, h10, disc, ITR (str): promoter sequences required in the output

    Returns:
    - List of dictionaries with the results of each predictions. Each dictionary contains:
        - the predicted value (float),
        - the difference with the target (float),
        - the promoter sequences for UP, h35, spacs, h10, disc, ITR (str[])

    """
    
    try:
        # Update "None" values in parameters with all valid lengths
        updated_params = remove_none_params(locals())

        # Load model
        model = load_model(model_path)

        # Create all permutations of the sequences
        all_sequences = get_sequences(max_iter, updated_params)

        # Run the model on all sequences
        difference = run_models(model, target, tolerance, all_sequences)

    except Exception as e:
        return f"An exception occurred: {e}"
    
    # return the top "results" results
    return list(difference.values())[:max_results]


# Update None-value parameters, replace with all possible combinations
def remove_none_params(params):
    # Define all valid lengths of each sequence
    lengths = {
        'UP': [16, 20, 22],
        'h35': [6],
        'spacs': [15, 16, 17, 18, 19],
        'h10': [6],
        'disc': [8, 6, 7],
        'ITR': [20, 21]
    }

    # Update parameters
    updated_params = {}
    for param_name, param_lengths in lengths.items():
        if params[param_name] is None:
            for l in param_lengths:
                updated_params[param_name] = [list(s) for s in list(combinations_with_replacement('ACTG', l))]
                break
        else:
            updated_params[param_name] = [params[param_name]]
    
    return updated_params

    
# Generate all possible combinations of sequences
def get_sequences(max_iter, updated_params):

    all_sequences = np.empty((0, 6), dtype=object)
    num_all_permutations = calc_num_permutations(updated_params)
    num_calc_permutations = 0

    for UP_seq in updated_params['UP']:
        for h35_seq in updated_params['h35']:
            for spacs_seq in updated_params['spacs']:
                for h10_seq in updated_params['h10']:
                    for disc_seq in updated_params['disc']:
                        for ITR_seq in updated_params['ITR']:
                            
                            # Add max_iter to avoid long runtimes
                            num_calc_permutations += 1
                            if num_calc_permutations > max_iter:
                                print(f"The maximum number of iterations has been reached. Simulating {max_iter}/{num_all_permutations} permutations.")
                                return all_sequences

                            # Concatenate sequences, append to array
                            seq = np.array(["".join(s) if isinstance(s, list) else s for s in [UP_seq, h35_seq, spacs_seq, h10_seq, disc_seq, ITR_seq]])
                            all_sequences = np.append(all_sequences, [seq], axis=0)
    
    return all_sequences


# Calculate the total number of permutations
def calc_num_permutations(updated_params):
    return reduce(lambda x, y: x * y, [len(updated_params[key]) for key in ['UP', 'h35', 'spacs', 'h10', 'disc', 'ITR']])


# Run the model on all sequences
def run_models(model, target, tolerance, all_sequences):
    difference = SortedDict()
    length = len(all_sequences)
    for i, seq in enumerate(all_sequences):

        # print progress, suppress model output
        print(f"Simulating sequence {i+1}/{length}", end='\r')
        with SuppressOutput():
            rate = model.predict(one_hot(seq))[0][0]

        # add to dictionary if the difference is within the tolerance
        if abs(rate) - abs(target) <= tolerance:
            difference[abs(rate) - abs(target)] = {'Predicted log(TX/Txref)' : rate,
                                            'Difference' : abs(rate) - abs(target),
                                            'UP' : seq[0],
                                            'h35' : seq[1],
                                            'spacs' : seq[2],
                                            'h10' : seq[3],
                                            'disc' : seq[4],
                                            'ITR' : seq[5]
                                            }
    # print progress, suppress model output
    print(f"Simulation complete: {i+1}/{length}", end='\r')
    return difference

# encode one row, concatenating each sequence
def one_hot(feature):
    max_len = {0 : 22, 1 : 6, 2 : 18, 3 : 6, 4 : 8, 5 : 21} # max length of each column
    concatenate = []
    for i in range(len(feature)):
        concatenate += padded_one_hot_encode('0' * (max_len[i] - len(feature[i])) + feature[i])
    return np.array([concatenate])
        
# encodes one sequence (one row at one column)
def padded_one_hot_encode(sequence):
    mapping = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0], 'T': [0,0,0,1,0], '0': [0,0,0,0]}
    encoding = []
    for nucleotide in sequence:
         encoding += [mapping[nucleotide]]
    return encoding


seq = ['TTTTCTATCTACGTAC', 'TTGACA', 'CTATTTCCTATTTCTCT', 'TATAAT', 'CCCCGCGG', 'CTCTACCTTAGTTTGTACGTT']
model_path = 'models/Hyperparameter_tuned.keras'

rate = pred_trans(seq, model_path)
print(f'Predicted transcrition rate (log(TX/Txref)): {rate}')