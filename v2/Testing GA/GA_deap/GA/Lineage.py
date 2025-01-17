import random
from deap import tools  # type: ignore

class Lineage:
    def __init__(self, toolbox, population_size, crossover_rate, mutation_prob, reconstruct_sequence, reverse_one_hot_sequence, cnn, elitism_rate, survival_rate):
        """
        Lineage initialization.
        """
        self.toolbox = toolbox
        self.population_size = population_size
        self.generation_idx = 0
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob
        self.reconstruct_sequence = reconstruct_sequence
        self.reverse_one_hot_sequence = reverse_one_hot_sequence
        self.cnn = cnn
        self.elitism_rate = elitism_rate
        self.survival_rate = survival_rate
        
        self.population = self.toolbox.population(n=self.population_size)
        self.best_sequence = None
        self.best_fitness = None
        self.best_prediction = None

    def run(self, generations=1):
        """
        Run the Genetic Algorithm for this lineage for a given number of generations.
        """
        # Evaluate initial population
        fitnesses = self.toolbox.evaluate(self.population)
        for individual, fit in zip(self.population, fitnesses):
            individual.fitness.values = fit

        # Track the initial best individual
        best_individual = tools.selBest(self.population, 1)[0]
        self._update_best(best_individual)

        # Start evolution
        for _ in range(generations):
            self.generation_idx += 1

            # Apply Elitism
            elite_size = int(self.elitism_rate * self.population_size)
            elite = tools.selBest(self.population, elite_size)

            # Select parents to mate (excluding elite)
            surviving_n = int(self.survival_rate * self.population_size) - elite_size
            parents = self.toolbox.select(self.population, surviving_n)
            random.shuffle(parents)

            # Generate offspring until the remaining slots are filled
            offspring = []
            i = 0
            while len(offspring) < self.population_size - elite_size:
                parent1 = parents[i % len(parents)]
                parent2 = parents[(i + 1) % len(parents)]
                child1, child2 = map(self.toolbox.clone, (parent1, parent2))
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                offspring.extend([child1, child2])
                i += 1

            # Apply Mutation
            for individual in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(individual, generation_idx=self.generation_idx)
                    del individual.fitness.values

            # Evaluate offspring
            invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_individuals:
                fitnesses = self.toolbox.evaluate(invalid_individuals)
                for individual, fit in zip(invalid_individuals, fitnesses):
                    individual.fitness.values = fit

            # Combine elite and offspring to form the new population
            self.population[:] = elite + offspring[:self.population_size - elite_size]

            # Update the best individual
            current_best = tools.selBest(self.population, 1)[0]
            self._update_best(current_best)


    def _update_best(self, individual):
        if self.best_fitness is None or individual.fitness.values[0] > self.best_fitness:
            self.best_fitness = individual.fitness.values[0]
            reconstructed_sequence = self.reconstruct_sequence(individual)
            self.best_sequence = self.reverse_one_hot_sequence(reconstructed_sequence)
            self.best_prediction = self.cnn.predict([reconstructed_sequence])[0]
