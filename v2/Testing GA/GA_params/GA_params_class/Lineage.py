import random

from Island import Island


class Lineage:
    '''
    Each lineage consists is independent and consists of multiple islands, each of which runs a genetic algorithm to optimize the target expression.
    Lineages are intended to explore different evolutionary paths to the target expression. This class also controls gene flow between islands.
    
    '''
    def __init__(self, geneticAlgorithm, lineage_idx):
        self.geneticAlgorithm = geneticAlgorithm
        self.lineage_idx = lineage_idx
        self.generation_idx = 0
        self.islands = [Island(self, geneticAlgorithm, island_idx) for island_idx in range(geneticAlgorithm.islands)]

        self.best_infill = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

    def run(self):
        while self.generation_idx < self.geneticAlgorithm.generations:
            for island in self.islands:
                island.population = island.generate_next_population()

            if self.geneticAlgorithm.islands > 1 and self.geneticAlgorithm.gene_flow_rate > 0:
                self.apply_gene_flow()

            self.update_best()

            if self.check_early_stopping():
                break

            self.generation_idx += 1

        return self.best_infill, self.best_prediction  
    
    def apply_gene_flow(self):        
        for recipient_idx in range(self.islands):
            # Select a random island to exchange individuals with
            donor_idx = random.choice([j for j in range(self.islands) if j != recipient_idx])
            num_individuals = int(self.geneticAlgorithm.gene_flow_rate * len(self.islands[recipient_idx].population))

            # Select individuals to migrate
            migrants_to_island = random.sample(self.islands[donor_idx].population, num_individuals)
            migrants_from_island = random.sample(self.islands[recipient_idx].population, num_individuals)

            # Exchange individuals
            self.islands[recipient_idx].population.extend(migrants_to_island)
            self.islands[donor_idx].population.extend(migrants_from_island)

            # Ensure pop do not exceed original size
            self.islands[recipient_idx].population = random.sample(self.islands[recipient_idx].population, len(self.islands[recipient_idx].population) - num_individuals)
            self.islands[donor_idx].population = random.sample(self.islands[donor_idx].population, len(self.islands[donor_idx].population) - num_individuals)

    def update_best(self):
        for island in self.islands:
            if island.best_fitness > self.best_fitness:
                self.best_fitness = island.best_fitness
                self.best_infill = island.best_infill
                self.best_prediction = island.best_prediction

    def check_early_stopping(self):
        if self.geneticAlgorithm.precision == None:
            return False
        if abs(self.best_prediction - self.geneticAlgorithm.target_expression) < self.geneticAlgorithm.precision:
            if self.geneticAlgorithm.verbose > 0:
                print(f'Lineage {self.idx+1}: Early stopping as target TX rate is achieved.')
            return True
        return False             