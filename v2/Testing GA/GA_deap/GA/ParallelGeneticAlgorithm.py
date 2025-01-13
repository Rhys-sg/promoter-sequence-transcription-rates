import random
import math
import numpy as np
import torch
import tensorflow as tf
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from deap import base, creator, tools  # type: ignore

from .Lineage import Lineage
from .CNN import CNN

from .Operators.CrossoverMethod import CrossoverMethod
from .Operators.MutationMethod import MutationMethod
from .Operators.SelectionMethod import SelectionMethod

class ParallelGeneticAlgorithm:
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
            selection_method='selTournament',
            boltzmann_temperature=0.5,
            tournsize=5,
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
        self.selection_method = getattr(SelectionMethod(boltzmann_temperature, tournsize), selection_method)

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
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._parallel_evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self.crossover_method)
        self.toolbox.register("mutate", self.mutation_method)

    def _evaluate_population(self, population, cnn, masked_sequence, target_expression, use_cache, reconstruct_sequence):
        reconstructed_sequences = [reconstruct_sequence(ind) for ind in population]
        predictions = cnn.predict(reconstructed_sequences, use_cache=use_cache)
        fitness = 1 - abs(target_expression - predictions)
        return [(fit,) for fit in fitness]

    def _generate_nucleotide(self):
        nucleotide = [0, 0, 0, 0]
        nucleotide[random.randint(0, 3)] = 1
        return tuple(nucleotide)

    def _generate_individual(self):
        return [self._generate_nucleotide() for _ in range(len(self.mask_indices))]

    def _parallel_evaluate(self, population):
        with Pool(processes=cpu_count()) as pool:
            # Split population into chunks for parallel processing
            chunk_size = max(1, len(population) // cpu_count())
            chunks = [population[i:i + chunk_size] for i in range(0, len(population), chunk_size)]

            # Create a partial function for _evaluate_population with fixed arguments
            evaluate_partial = partial(
                self._evaluate_population,
                cnn=self.cnn,
                masked_sequence=self.masked_sequence,
                target_expression=self.target_expression,
                use_cache=self.use_cache,
                reconstruct_sequence=self._reconstruct_sequence
            )

            # Evaluate each chunk in parallel
            results = pool.map(evaluate_partial, chunks)

            # Flatten the results
            return [fitness for chunk_result in results for fitness in chunk_result]
        
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
