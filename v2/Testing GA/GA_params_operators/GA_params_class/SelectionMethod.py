import random
import math

class SelectionMethod():
    '''
    This class implements various selection methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, surviving_pop, elitist_rate, num_competitors, steady_state_k, boltzmann_temperature):
        self.elitist_rate = elitist_rate
        self.num_competitors = min(num_competitors, surviving_pop) # Ensure num_competitors is not larger than surviving_pop
        self.steady_state_k = steady_state_k
        self.boltzmann_temperature = boltzmann_temperature
    
    def tournament(self, population, fitness_scores, surviving_pop):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected.'''
        parents = self.truncation(population, fitness_scores, max(1, int(self.elitist_rate * surviving_pop))) if self.elitist_rate > 0 else []
        for _ in range(surviving_pop):
            competitors = random.sample(range(len(population)), k=self.num_competitors)
            winner = max(competitors, key=lambda idx: fitness_scores[idx])
            parents.append(population[winner])
        return parents
    
    def tournament_pop(self, population, fitness_scores, surviving_pop):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected and removed from future tournaments.'''
        remaining_population = list(population)
        remaining_fitness_scores = list(fitness_scores)
        parents = []
        for _ in range(surviving_pop):
            competitors = random.sample(range(len(remaining_population)), k=self.num_competitors)
            winner_idx = max(competitors, key=lambda idx: remaining_fitness_scores[idx])
            parents.append(remaining_population[winner_idx])
            del remaining_population[winner_idx]
            del remaining_fitness_scores[winner_idx]
        return parents
    
    def roulette(self, population, fitness_scores, surviving_pop):
        ''''Individuals are selected with a probability proportional to their fitness.'''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = []
        for _ in range(surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def linear_scaling(self, population, fitness_scores, surviving_pop):
        '''Fitness scores are normalized, and then roulette selection is performed.'''
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        adjusted_scores = [(score - min_fitness) / (max_fitness - min_fitness + 1e-6) for score in fitness_scores]
        return self.roulette(population, adjusted_scores, surviving_pop)
    
    def rank_based(self, population, fitness_scores, surviving_pop):
        '''Individuals are ranked based on their fitness, and selection probabilities are assigned based on rank rather than absolute fitness.'''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx])
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        total_rank = sum(ranks.values())
        probabilities = [ranks[idx] / total_rank for idx in range(len(population))]
        parents = []
        for _ in range(surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def sus(self, population, fitness_scores, surviving_pop):
        '''
        Similar to roulette wheel selection, but instead of selecting one individual at a time,
        Stochastic Universal Sampling (SUS) uses multiple equally spaced pointers to select individuals simultaneously.

        '''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        step = 1.0 / surviving_pop
        start = random.uniform(0, step)
        pointers = [start + i * step for i in range(surviving_pop)]
        
        parents = []
        for pointer in pointers:
            for idx, cumulative in enumerate(cumulative_probabilities):
                if pointer <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def truncation(self, population, fitness_scores, surviving_pop, reverse=False):
        '''
        Only the top individuals are selected for the next generation.
        This method is reused for elitist selection by setting elitist_rate to a value between 0 and 1.
        '''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx], reverse=True)
        if reverse:
            sorted_indices = sorted_indices[::-1]
        parents = [population[idx] for idx in sorted_indices[:surviving_pop]]
        return parents
    
    def boltzmann(self, population, fitness_scores, surviving_pop):
        '''
        Based on simulated annealing, this method adjusts selection probabilities dynamically over time,
        favoring exploration in early generations and exploitation in later generations.
        
        '''
        boltzmann_scores = [math.exp(score / self.boltzmann_temperature) for score in fitness_scores]
        total_score = sum(boltzmann_scores)
        probabilities = [score / total_score for score in boltzmann_scores]
        parents = []
        for _ in range(surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents