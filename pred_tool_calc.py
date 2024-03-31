from keras.saving import load_model
import numpy as np
from itertools import combinations_with_replacement
from sortedcontainers import SortedDict
import sys
import os

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
        seq = one_hot(seq)
        model = load_model(model_path)
        rate = model.predict(seq)
    except Exception as e:
        return f"An exception occurred: {e}"
    return rate


def pred_prom(model_path, target, tolerance=float('inf'), max_results=5, max_iter=100,
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
    params = {'UP': [16, 20, 22],
            'h35': [6],
            'spacs': [16, 17, 18],
            'h10': [6],
            'disc': [8, 6, 7],
            'ITR': [20, 21]
    }

    # Update "None" values in parameters with all possible combinations
    updated_params = {}
    for param_name, param_lengths in params.items():
        if locals()[param_name] is None:
            for l in param_lengths:
                updated_params[param_name] = [list(s) for s in list(combinations_with_replacement('ACTG', l))]
                break
        else:
            updated_params[param_name] = [locals()[param_name]]

    # Load model
    model = load_model(model_path)

    # Create all permutations of the sequences
    all_sequences = get_sequences(max_iter, updated_params)

    # Run the model on all sequences
    difference = run_models(model, target, tolerance, all_sequences)

    # return the top "results" results
    return list(difference.values())[:max_results]

    
# Generate all possible combinations of sequences
def get_sequences(max_iter, updated_params):
    all_sequences = []
    for UP_seq in updated_params['UP']:
        for h35_seq in updated_params['h35']:
            for spacs_seq in updated_params['spacs']:
                for h10_seq in updated_params['h10']:
                    for disc_seq in updated_params['disc']:
                        for ITR_seq in updated_params['ITR']:
                            
                            # Add max_iter to avoid long runtimes
                            max_iter -= 1
                            if max_iter <= 0: 
                                print("Max iterations reached. Truncating results.")
                                return all_sequences

                            # Concatenate sequences, append to list
                            seq = np.array(["".join(s) if isinstance(s, list) else s for s in [UP_seq, h35_seq, spacs_seq, h10_seq, disc_seq, ITR_seq]])
                            all_sequences += [seq]
    
    return all_sequences


# Run the model on all sequences
def run_models(model, target, tolerance, all_sequences):
    difference = SortedDict()
    length = len(all_sequences)
    for i, seq in enumerate(all_sequences):

        # print progress, suppress model output
        print(f"Simulating sequence {i+1}/{length}", end='')
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
    mapping = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 'G': [0,0,1,0,0], 'T': [0,0,0,1,0], '0': [0,0,0,0,1]}
    encoding = []
    for nucleotide in sequence:
         encoding += [mapping[nucleotide]]
    return encoding