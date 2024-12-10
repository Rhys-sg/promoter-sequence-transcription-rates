import torch
import numpy as np
import random
import re
import math
from keras.models import load_model  # type: ignore

from Lineage import Lineage
from SelectionMethod import SelectionMethod

class GeneticAlgorithm:
    '''
    This class is the interface for the genetic algorithm.
    
    The algorithm infills a masked sequence with nucleotides that maximize the predicted transcription rate.
    For each lineage, the algorithm evolves a population of sequences over multiple generations. A population consists of multiple islands
    that evolve independently with occasional gene flow between them. 
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using the specified selection method, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    '''

    def __init__(
            self,
            cnn_model_path,
            masked_sequence,
            target_expression,
            precision=None,
            max_length=150,
            pop_size=100,
            generations=100, 
            base_mutation_rate=0.05,
            chromosomes=1,
            elitist_rate=0,
            previous_lineage_hamming_alpha=1,
            islands=1,
            gene_flow_rate=0,
            surval_rate=0.5,
            num_parents=2,
            num_competitors=5,
            selection='tournament',
            boltzmann_temperature=1,
            verbose=1,
            seed=None
    ):
        self.device = self.get_device()
        self.cnn = load_model(cnn_model_path)
        self.masked_sequence = masked_sequence
        self.target_expression = target_expression
        self.precision = precision
        self.max_length = max_length
        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = base_mutation_rate
        self.chromosomes = chromosomes
        self.elitist_rate = elitist_rate
        self.previous_lineage_hamming_alpha = previous_lineage_hamming_alpha
        self.islands = islands
        self.gene_flow_rate = gene_flow_rate
        self.surviving_pop = max(1, int((self.pop_size / self.islands) * surval_rate)) # Ensure surviving_pop is at least 1
        self.num_parents = min(num_parents, self.surviving_pop) # Ensure num_parents is not larger than surviving_pop
        self.selection_method = getattr(SelectionMethod(self.surviving_pop, elitist_rate, num_competitors, boltzmann_temperature), selection)
        self.verbose = verbose
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.chromosome_lengths = self.split_chromosome_lengths(self.mask_length, chromosomes)

        # For tracking and memoization purposes, could use lru_cache instead
        self.previous_lineage_infills = {}
        self.seen_infills = {}

        # Store the best sequences and predictions for each lineage
        self.best_infills = []
        self.best_predictions = []

        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def split_chromosome_lengths(self, total_length, chromosomes):
        '''Split the mask length into chromosome lengths.'''
        base_length = total_length // chromosomes
        lengths = [base_length] * chromosomes
        for i in range(total_length % chromosomes):
            lengths[i] += 1
        return lengths
    
    def run(self, lineages=1):
        '''Run the genetic algorithm for the specified number of lineages.'''
        for lineage_idx in range(lineages):
            lineage = Lineage(self, lineage_idx)

            # Run the genetic algorithm for the current lineage
            best_infill, best_prediction = lineage.run()

            # Update the seen infills with the best infill from the current lineage
            self.previous_lineage_infills.update(self.seen_infills)

            self.best_infills.append(best_infill)
            self.best_predictions.append(best_prediction)

            self.print_progress(lineage_idx, best_infill, best_prediction)

        return self.best_infills, self.best_predictions
    
    def print_progress(self, lineage_idx, infill, best_prediction):
        if self.verbose > 0:
            best_sequence = self.reconstruct_sequence(infill)
            print(f'Lineage {lineage_idx+1} Complete: Best TX rate: {best_prediction:.4f} | Best Sequence: {best_sequence}')

    def reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return ''.join(sequence)


if __name__ == '__main__':
    cnn_model_path = 'v2/Models/CNN_6_1_2.keras'
    masked_sequence = 'AATACTAGAGGTCTTCCGACNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGTGTGGGCGGGAAGACAACTAGGGG'
    target_expression = 1

    ga = GeneticAlgorithm(
        cnn_model_path=cnn_model_path,
        masked_sequence=masked_sequence,
        target_expression=target_expression,
    )
    best_sequences, best_predictions = ga.run(3)
    print('\nBest infilled sequences:', best_sequences)
    print('Predicted transcription rates:', best_predictions)