import random
from deap import tools  # type: ignore

class Lineage:
    def __init__(self, toolbox, population_size, generations, crossover_rate, mutation_prob, reconstruct_sequence, reverse_one_hot_sequence, cnn):
        self.toolbox = toolbox
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_prob = mutation_prob
        self.reconstruct_sequence = reconstruct_sequence
        self.reverse_one_hot_sequence = reverse_one_hot_sequence
        self.cnn = cnn
        
        self.population = self.toolbox.population(n=self.population_size)
        self.best_sequence = None
        self.best_fitness = None
        self.best_prediction = None
    
    def run(self):
        """Run the Genetic Algorithm for this lineage."""
        # Evaluate initial population
        fitnesses = self.toolbox.evaluate(self.population)
        for individual, fit in zip(self.population, fitnesses):
            individual.fitness.values = fit

        # Start evolution
        for generation_idx in range(self.generations):
            offspring = self.toolbox.select(self.population, len(self.population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply Mutation
            for individual in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(individual, generation_idx=generation_idx)
                    del individual.fitness.values

            # Evaluate offspring
            invalid_individual = [individual for individual in offspring if not individual.fitness.valid]
            if invalid_individual:
                fitnesses = self.toolbox.evaluate(invalid_individual)
                for individual, fit in zip(invalid_individual, fitnesses):
                    individual.fitness.values = fit

            # Replace population
            self.population[:] = offspring

        # Get the best individual
        best_individual = tools.selBest(self.population, 1)[0]
        reconstructed_sequence = self.reconstruct_sequence(best_individual)
        self.best_sequence = self.reverse_one_hot_sequence(reconstructed_sequence)
        self.best_fitness = best_individual.fitness.values[0]
        self.best_prediction = self.cnn.predict([reconstructed_sequence])[0]
    

