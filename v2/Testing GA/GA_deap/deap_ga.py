import random
import numpy as np
import math
from deap import base, creator, tools  # type: ignore

from cnn import CNN

class GeneticAlgorithm:
    def __init__(self, cnn_model_path, use_cache, masked_sequence, target_expression, population_size, generations, crossover_prob, mutation_prob):
        # Genetic Algorithm attributes
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # CNN model and attributes
        self.cnn = CNN(cnn_model_path)
        self.use_cache = use_cache

        # Evaluation attributes
        self.masked_sequence = self.cnn.one_hot_sequence(masked_sequence)
        self.mask_indices = self._get_mask_indices(self.masked_sequence)
        self.target_expression = target_expression

        # Setup DEAP
        self.toolbox = base.Toolbox()
        self._setup_deap()

        # Cache for infill sequences
        self.infill_cache = {}

    def _get_mask_indices(self, masked_sequence):
        return [i for i, element in enumerate(masked_sequence) if all(math.isclose(e, 0.25, rel_tol=1e-9) for e in element)]

    def _setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        def generate_one_hot():
            nucleotide = np.array([0, 0, 0, 0])
            nucleotide[random.randint(0, 3)] = 1
            return nucleotide

        def generate_individual():
            return [generate_one_hot() for _ in range(len(self.mask_indices))]

        # Batch evaluation
        def eval_fitness_batch(population):
            population = [self._reconstruct_sequence(ind) for ind in population]
            predictions = self.cnn.predict(population, use_cache=self.use_cache)
            fitnesses = 1 - np.abs(predictions - self.target_expression)
            return [(fit,) for fit in fitnesses]
        
        # Override map to process individuals in batches
        def batch_map(evaluate, individuals):
            return evaluate(individuals)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", eval_fitness_batch)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", self._cx_one_hot)
        self.toolbox.register("mutate", self._mut_one_hot)

        self.toolbox.register("map", batch_map)
        
    def _reconstruct_sequence(self, infill):
        sequence = list(self.masked_sequence)
        for idx, char in zip(self.mask_indices, infill):
            sequence[idx] = char
        return sequence

    def _cx_one_hot(self, ind1, ind2):
        """Crossover: Swap nucleotides between individuals."""
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]

    def _mut_one_hot(self, individual):
        """Mutation: Replace one nucleotide with a new one-hot nucleotide."""
        nucleotide_idx = random.randint(0, len(individual) - 1)
        nucleotide = np.array([0, 0, 0, 0])
        nucleotide[random.randint(0, 3)] = 1
        individual[nucleotide_idx] = nucleotide
        return (individual,)

    def run(self):
        """Run the Genetic Algorithm."""
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = self.toolbox.evaluate(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Start evolution
        for gen in range(self.generations):
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = self.toolbox.evaluate(invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # Replace population
            population[:] = offspring

        # Return the best individual
        best_ind = tools.selBest(population, 1)[0]
        reconstructed_sequence = self._reconstruct_sequence(best_ind)
        best_sequence = self.cnn.reverse_one_hot_sequence(reconstructed_sequence)
        prediction = self.cnn.predict([reconstructed_sequence])
        return best_sequence, best_ind.fitness.values, prediction
