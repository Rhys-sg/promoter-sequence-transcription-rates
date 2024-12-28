import torch
import numpy as np
import random
import re
import math
from keras.models import load_model  # type: ignore

from .Lineage import Lineage
from .SelectionMethod import SelectionMethod
from .MutationMethod import MutationMethod
from .CrossoverMethod import CrossoverMethod
from .ParentChoiceMethod import ParentChoiceMethod

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

            # Search parameters
            pop_size=100,
            generations=150,
            survival_rate=0.5,

            # MOGA parameters
            lineage_divergence_alpha=0,
            diversity_alpha=0,

            # Parallel GA/island parameters
            islands=1,
            gene_flow_rate=0,

            # selection parameters
            selection='tournament',
            boltzmann_temperature=0.03,
            elitist_rate=0,
            steady_state_k=2,
            num_competitors=5,

            # mutation parameters
            mutation='relative_bit_string',
            mutation_rate=1,
            relative_mutation_rate_alpha=1,

            # crossover parameters
            crossover='single_point',
            k_crossover_points=2,

            # parent choice parameters
            parent_choice='by_order',
            covariance=0,
            generational_covariance_alpha=0,

            # other parameters
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
        self.survival_rate = survival_rate
        self.lineage_divergence_alpha = lineage_divergence_alpha
        self.diversity_alpha = diversity_alpha
        self.islands = islands
        self.gene_flow_rate = gene_flow_rate
        self.island_pop = max(1, int((self.pop_size / self.islands))) # Ensure it is at least 1

        # Operators and their parameters
        self.selection_method = getattr(SelectionMethod(self.island_pop, boltzmann_temperature, steady_state_k, num_competitors), selection)
        self.elitist_rate = elitist_rate
        self.steady_state_k = steady_state_k
        self.mutation_method = getattr(MutationMethod(mutation_rate, generations, relative_mutation_rate_alpha), mutation)
        self.crossover_method = getattr(CrossoverMethod(k_crossover_points), crossover)
        self.parent_choice_method = getattr(ParentChoiceMethod(covariance, generational_covariance_alpha), parent_choice)

        self.verbose = verbose
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)

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

        # track lineages
        self.lineages = []

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def run(self, lineages=1):
        '''Run the genetic algorithm for the specified number of lineages.'''
        for lineage_idx in range(lineages):
            lineage = Lineage(self, lineage_idx)
            self.lineages.append(lineage)

            # Run the genetic algorithm for the current lineage
            best_infill, best_prediction = lineage.run()

            # Update the seen infills with the best infill from the current lineage
            self.previous_lineage_infills.update(self.seen_infills)

            self.best_infills.append(best_infill)
            self.best_predictions.append(best_prediction)

            self.print_progress(lineage_idx, best_infill, best_prediction)

        return [self.reconstruct_sequence(infill) for infill in self.best_infills], self.best_predictions
    
    def print_progress(self, lineage_idx, infill, best_prediction):
        if self.verbose > 0:
            best_sequence = self.reconstruct_sequence(infill)
            print(f'Lineage {lineage_idx+1} Complete: Best TX rate: {best_prediction:.4f} | Best Sequence: {best_sequence}')

    def reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return ''.join(sequence)
    
    def get_infill_history(self):
        return [lineage.get_infill_history() for lineage in self.lineages]
    
    def get_fitness_history(self):
        return [lineage.get_fitness_history() for lineage in self.lineages]
    
    def get_prediction_history(self):
        return [lineage.get_prediction_history() for lineage in self.lineages]