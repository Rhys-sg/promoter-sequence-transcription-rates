import random
from deap import tools  # type: ignore

class Lineage:
    def __init__(self, toolbox, population_size, generations, crossover_prob, mutation_prob, reconstruct_sequence, reverse_one_hot_sequence, cnn):
        self.toolbox = toolbox
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
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
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit

        # Start evolution
        for gen in range(self.generations):
            offspring = self.toolbox.select(self.population, len(self.population))
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
            self.population[:] = offspring

        # Get the best individual
        best_ind = tools.selBest(self.population, 1)[0]
        reconstructed_sequence = self.reconstruct_sequence(best_ind)
        self.best_sequence = self.reverse_one_hot_sequence(reconstructed_sequence)
        self.best_fitness = best_ind.fitness.values[0]
        self.best_prediction = self.cnn.predict([reconstructed_sequence])[0]
    

