import random
import math
import numpy as np
import torch
import tensorflow as tf
import os
from deap import base, creator, tools  # type: ignore

from .Lineage import Lineage
from .CNN import CNN

from .Operators.CrossoverMethod import CrossoverMethod
from .Operators.MutationMethod import MutationMethod
from .Operators.SelectionMethod import SelectionMethod

class GeneticAlgorithm:
    '''
    For actual use, refer to the GeneticAlgorithm class in the GA module.

    This is the test version of the GeneticAlgorithm class. It is used to test genetic operators, their parameters, and the DEAP library.
    This version does not implement the Coarse-Grained Parallel GA or MOGA. For those, refer to the MogaGeneticAlgorithm.py and ParallelGeneticAlgorithm.py

    This class is used to run multiple lineages of the Genetic Algorithm, each simulating an independent attempt to find a sequence that minimizes the
    difference between the target expression and the CNN's prediction. GAs are stochastic, so multiple lineages are run to reduce the effect of randomness and
    show overall trends.

    '''
    def __init__(
            self,
            cnn_model_path,
            masked_sequence,
            target_expression,
            use_cache=True,
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
            selection_method='selBest',
            boltzmann_temperature=0.1,
            tournsize=5,

            # Additional parameters
            elitism_rate=0,
            survival_rate=0.5,
    ):
        # Set seed
        if seed is not None:
            self._set_seed(seed)

        # Genetic Algorithm attributes
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob

        # CNN model and attributes
        self.cnn = CNN(cnn_model_path)
        self.use_cache = use_cache

        # Evaluation attributes
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression

        # Operators
        self.adj_mutation_rate = getattr(MutationMethod(mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations), mutation_method)
        self.crossover_method = getattr(CrossoverMethod(crossover_points), crossover_method)
        self.selection_method = getattr(SelectionMethod(boltzmann_temperature, tournsize), selection_method)

        # Additional parameters
        self.elitism_rate = elitism_rate
        self.survival_rate = survival_rate

        # Setup DEAP
        self.toolbox = base.Toolbox()
        self._setup_deap()

        # Lineage objects
        self.lineage_objects = []

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
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        def generate_nucleotide():
            nucleotide = [0, 0, 0, 0]
            nucleotide[random.randint(0, 3)] = 1
            return tuple(nucleotide)

        def generate_individual():
            return [generate_nucleotide() for _ in range(len(self.mask_indices))]

        def evaluate(population):
            population = [self._reconstruct_sequence(ind) for ind in population]
            predictions = self.cnn.predict(population, use_cache=self.use_cache)
            fitness = 1 - abs(self.target_expression - predictions)
            return [(fit,) for fit in fitness]
    
        # Override map to process individuals in batches
        def batch_map(evaluate, individuals):
            return evaluate(individuals)

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

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("select", self.selection_method)
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
            lineage = Lineage(
                toolbox=self.toolbox,
                population_size=self.population_size,
                crossover_rate=self.crossover_rate,
                mutation_prob=self.mutation_prob,
                reconstruct_sequence=self._reconstruct_sequence,
                reverse_one_hot_sequence=self.cnn.reverse_one_hot_sequence,
                cnn=self.cnn,
                elitism_rate=self.elitism_rate,
                survival_rate=self.survival_rate,
            )
            
            lineage.run(self.generations)
            self.lineage_objects.append(lineage)
    
    @property
    def best_sequences(self):
        return [lineage.best_sequence for lineage in self.lineage_objects]
    
    @property
    def best_fitnesses(self):
        return [lineage.best_fitness for lineage in self.lineage_objects]
    
    @property
    def best_predictions(self):
        return [lineage.best_prediction for lineage in self.lineage_objects]
