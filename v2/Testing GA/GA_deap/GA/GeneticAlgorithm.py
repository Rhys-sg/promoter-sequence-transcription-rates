import random
import math
import numpy as np
import torch
import tensorflow as tf
import os
from deap import base, creator, tools  # type: ignore

from .CNN import CNN

class GeneticAlgorithm:
    def __init__(
            self,
            cnn_model_path,
            masked_sequence,
            target_expression,
            use_cache=True,
            population_size=100,
            generations=100,
            seed=None,
    ):
        # Set seed
        if seed is not None:
            self._set_seed(seed)

        # Genetic Algorithm attributes
        self.population_size = population_size
        self.generations = generations

        # CNN model and attributes
        self.cnn = CNN(cnn_model_path)
        self.use_cache = use_cache

        # Evaluation attributes
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression

        # Mutation attributes
        self.mutation_rate = 0.1
        self.mutation_prob = 0.6

        # Setup DEAP
        self.toolbox = base.Toolbox()
        self._setup_deap()

        # Initialize population
        self.population = self.toolbox.population(n=self.population_size)

        # Best individual attributes
        self.best_sequence = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

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

        def mutConstant(individual):
            '''Each nucleotide in the bit string has a probability of mutating.'''
            for i in range(len(individual)):
                if random.random() < self.mutation_rate:
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
        self.toolbox.register("select", tools.selBest)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", mutConstant)

        self.toolbox.register("map", batch_map)
        
    def _reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return sequence

    def run(self):
        """
        Run the Genetic Algorithm.
        """
        # Evaluate initial population
        fitnesses = self.toolbox.evaluate(self.population)
        for individual, fit in zip(self.population, fitnesses):
            individual.fitness.values = fit

        # Track the initial best individual
        best_individual = tools.selBest(self.population, 1)[0]
        self._update_best(best_individual)

        # Start evolution
        for _ in range(self.generations):

            parents = self.toolbox.select(self.population, self.population_size)
            random.shuffle(parents)

            # Generate offspring until the remaining slots are filled
            offspring = []
            i = 0
            while len(offspring) < self.population_size:
                parent1 = parents[i % len(parents)]
                parent2 = parents[(i + 1) % len(parents)]
                child1, child2 = map(self.toolbox.clone, (parent1, parent2))
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
                offspring.extend([child1, child2])
                i += 1

            # Apply Mutation
            for individual in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(individual)
                    del individual.fitness.values

            # Evaluate offspring
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_individuals:
                fitnesses = self.toolbox.evaluate(invalid_individuals)
                for individual, fit in zip(invalid_individuals, fitnesses):
                    individual.fitness.values = fit

            # Update the best individual
            current_best = tools.selBest(self.population, 1)[0]
            self._update_best(current_best)


    def _update_best(self, individual):
        if self.best_fitness is None or individual.fitness.values[0] > self.best_fitness:
            self.best_fitness = individual.fitness.values[0]
            reconstructed_sequence = self._reconstruct_sequence(individual)
            self.best_sequence = self.cnn.reverse_one_hot_sequence(reconstructed_sequence)
            self.best_prediction = self.cnn.predict([reconstructed_sequence])[0]

