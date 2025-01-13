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
from .Operators.MogaSelectionMethod import MogaSelectionMethod

class MogaGeneticAlgorithm:
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
            mutation_prob=0.6,
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
            prediction_weight=1,
            divergence_weight=1,
            diversity_weight=1,

            # MOGA parameters
            divergence_method='max',
            diversity_method='mean'
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
        self.mutation_method = getattr(MutationMethod(mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations), mutation_method)
        self.crossover_method = getattr(CrossoverMethod(crossover_points), crossover_method)
        self.selection_method = getattr(MogaSelectionMethod(), selection_method)

        # MOGA attributes
        self.prediction_weight = prediction_weight
        self.divergence_weight = divergence_weight
        self.diversity_weight = diversity_weight
        moga_methods = {'max': max, 'min': min, 'mean': np.mean, 'std': np.std}
        self.divergence_method = moga_methods[divergence_method]
        self.diversity_method = moga_methods[diversity_method]

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
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(self.prediction_weight, self.divergence_weight, self.diversity_weight))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        def generate_nucleotide():
            nucleotide = [0, 0, 0, 0]
            nucleotide[random.randint(0, 3)] = 1
            return tuple(nucleotide)

        def generate_individual():
            return [generate_nucleotide() for _ in range(len(self.mask_indices))]
        
        def evaluate(population):
            prediction_fitnesses = prediction_fitness(population)
            divergence_fitnesses = divergence_fitness(population, self.divergence_method)
            diversity_fitnesses = diversity_fitness(population, self.diversity_method)
            return tuple(zip(prediction_fitnesses, divergence_fitnesses, diversity_fitnesses))

        def prediction_fitness(population):
            population = [self._reconstruct_sequence(ind) for ind in population]
            predictions = self.cnn.predict(population, use_cache=self.use_cache)
            return 1 - abs(predictions - self.target_expression)
        
        def divergence_fitness(population, method):
            if len(self.lineage_objects) == 0:
                return np.zeros(len(population))
            fitnesses = []
            for current_ind in population:
                fitnesses.append(1-method([hamming_distance(current_ind, previous_ind.best_sequence[0]) for previous_ind in self.lineage_objects]))
            return fitnesses
         
        def diversity_fitness(population, method):
            fitnesses = []
            for i, current_ind in enumerate(population, start=1):
                fitnesses.append(1-method([hamming_distance(current_ind, other_ind) for other_ind in population[:i:]]))
            return fitnesses
            
        def hamming_distance(ind1, ind2):
            return sum([1 for s, t in zip(ind1, ind2) if s != t]) / len(ind1)
    
        # Override map to process individuals in batches
        def batch_map(evaluate, individuals):
            return evaluate(individuals)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self.crossover_method)
        self.toolbox.register("mutate", self.mutation_method)

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
                generations=self.generations,
                crossover_rate=self.crossover_rate,
                mutation_prob=self.mutation_prob,
                reconstruct_sequence=self._reconstruct_sequence,
                reverse_one_hot_sequence=self.cnn.reverse_one_hot_sequence,
                cnn=self.cnn
            )
            
            lineage.run()
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
