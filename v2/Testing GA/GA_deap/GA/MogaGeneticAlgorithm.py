import random
import math
import numpy as np
import torch
import tensorflow as tf
import os
from deap import base, creator, tools  # type: ignore

from .MogaLineage import MogaLineage
from .CNN import CNN

from .Operators.CrossoverMethod import CrossoverMethod
from .Operators.MutationMethod import MutationMethod
from .Operators.MogaSelectionMethod import MogaSelectionMethod

class MogaGeneticAlgorithm:
    def __init__(
            self,
            masked_sequence,

            # Evaluation parameters
            evaluation_methods,
            evaluation_weights,

            # Genetic Algorithm parameters
            population_size=100,
            generations=100,
            seed=None,

            # Mutation parameters
            mutation_method='mutConstant',
            mutation_prob=0.2,
            mutation_rate=0.1,
            mutation_rate_start=0.1,
            mutation_rate_end=0.1,
            mutation_rate_degree=2,

            # Crossover parameters
            crossover_method='cxOnePoint',
            crossover_rate=1,
            crossover_points=2,

            # Selection parameters
            selection_method='selAutomaticEpsilonLexicase',

            # Additional parameters
            elitism_rate=0,
            elitism_selection_method=None,
            survival_rate=0.5,

            # Performance parameters
            track_history=False
    ):
        # Set seed
        if seed is not None:
            self._set_seed(seed)

        # Evaluation attributes
        self.evaluation_methods = evaluation_methods
        self.evaluation_weights = evaluation_weights

        # Genetic Algorithm attributes
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob

        # Evaluation attributes
        self.masked_sequence = CNN.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)

        # Operators
        self.adj_mutation_rate = getattr(MutationMethod(mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations), mutation_method)
        self.crossover_method = getattr(CrossoverMethod(crossover_points), crossover_method)
        self.selection_method = getattr(MogaSelectionMethod(), selection_method)
        if elitism_selection_method is None:
            elitism_selection_method = selection_method
        self.elitism_selection_method = getattr(MogaSelectionMethod(), elitism_selection_method)

        # Additional parameters
        self.elitism_rate = elitism_rate
        self.survival_rate = survival_rate

        # Setup DEAP
        self.toolbox = base.Toolbox()
        self._setup_deap()

        # Lineage objects
        self.lineage_objects = []

        # Evaluation parameters
        self.track_history = track_history

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if all(math.isclose(e, 0.25, rel_tol=1e-9) for e in element)]

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=self.evaluation_weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        def generate_nucleotide():
            nucleotide = [0, 0, 0, 0]
            nucleotide[random.randint(0, 3)] = 1
            return tuple(nucleotide)

        def generate_individual():
            return [generate_nucleotide() for _ in range(len(self.mask_indices))]
        
        def evaluate(infill_population):
            '''Evaluate the fitness of the population based on all evaluation_methods.'''
            reconstructed_population = [self._reconstruct_sequence(infill) for infill in infill_population]
            fitnesses = [
                evaluation_method(
                    infill_population=infill_population,
                    reconstructed_population=reconstructed_population
                )
                for evaluation_method in self.evaluation_methods
            ]
            return tuple(zip(*fitnesses))
        
        def mutate(individual, mutation_rate):
            '''The mutation rate remains constant over time.'''
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[i] = mutate_idv(individual[i])
            return (individual,)
        
        def mutate_idv(nucleotide):
            '''Randomly change a one-hot encoded nucleotide to another one-hot encoded nucleotide.'''
            nucleotide = [0, 0, 0, 0]
            nucleotide[random.randint(0, 3)] = 1
            return tuple(nucleotide)
    
        # Override map to process individuals in batches
        def batch_map(evaluate, individuals):
            return evaluate(individuals)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("select", self.selection_method)
        self.toolbox.register("select_elite", self.elitism_selection_method)
        self.toolbox.register("mate", self.crossover_method)
        self.toolbox.register("adj_mutation_rate", self.adj_mutation_rate)
        self.toolbox.register("mutate", mutate)
        self.toolbox.register("map", batch_map)
        
    def _reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return sequence

    def run(self, lineages=1):
        """Run multiple lineages of the Genetic Algorithm."""
        for lineage_id in range(lineages):
            lineage = MogaLineage(
                toolbox=self.toolbox,
                population_size=self.population_size,
                crossover_rate=self.crossover_rate,
                mutation_prob=self.mutation_prob,
                reconstruct_sequence=self._reconstruct_sequence,
                reverse_one_hot_sequence=CNN.reverse_one_hot_sequence,
                elitism_rate=self.elitism_rate,
                survival_rate=self.survival_rate,
                track_history=self.track_history,
            )
            
            lineage.run(self.generations)
            self.lineage_objects.append(lineage)
    
    '''
    Properties for best sequences, fitnesses, and predictions of each lineage
    '''
    @property
    def best_sequences(self):
        return [lineage.best_sequence for lineage in self.lineage_objects]
    
    @property
    def best_fitnesses(self):
        return [lineage.best_fitness for lineage in self.lineage_objects]
    
    @property
    def best_predictions(self):
        return [lineage.best_prediction for lineage in self.lineage_objects]
    
    '''
    Properties for the history of each lineage
    '''
    @property
    def population_history(self):
        return [lineage.population_history for lineage in self.lineage_objects]
    
    @property
    def best_sequence_history(self):
        return [lineage.best_sequence_history for lineage in self.lineage_objects]
    
    @property
    def best_fitness_history(self):
        return [lineage.best_fitness_history for lineage in self.lineage_objects]
    
    @property
    def best_prediction_history(self):
        return [lineage.best_prediction_history for lineage in self.lineage_objects]
    
    '''
    Properties for the convergence history of each lineage, the max, min, and mean convergence history of all lineages
    '''
    @property
    def convergence_history(self):
        return [lineage.convergence_history for lineage in self.lineage_objects]

    @property
    def max_lineage_convergence_history(self):
        convergence_history = self.convergence_history
        return [max([convergence_history[i][j] for i in range(len(convergence_history))]) for j in range(len(convergence_history[0]))]
    
    @property
    def min_lineage_convergence_history(self):
        convergence_history = self.convergence_history
        return [min([convergence_history[i][j] for i in range(len(convergence_history))]) for j in range(len(convergence_history[0]))]
    
    @property
    def mean_lineage_convergence_history(self):
        convergence_history = self.convergence_history
        return [np.mean([convergence_history[i][j] for i in range(len(convergence_history))]) for j in range(len(convergence_history[0]))]
    
    def reorder_history_by_generation(self, history):
        '''Reformats the history of each lineage to be ordered by generation, not lineage. Ensures all data is in a similar format.'''
        return [[history[i][j] for i in range(len(history))] for j in range(len(history[0]))]
