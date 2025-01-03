import random
import math
from collections import Counter

class SelectionMethod():
    '''
    This class implements various selection methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, surviving_N, boltzmann_temperature, steady_state_k, num_competitors):
        self.boltzmann_temperature = boltzmann_temperature
        self.steady_state_k = min(steady_state_k, surviving_N)
        self.num_competitors = min(num_competitors, surviving_N)

    def boltzmann(self, population, fitness_scores, surviving_N):
        '''
        Based on simulated annealing, this method adjusts selection probabilities dynamically over time,
        favoring exploration in early generations and exploitation in later generations.
        '''
        boltzmann_scores = [math.exp(score / self.boltzmann_temperature) for score in fitness_scores]
        total_score = sum(boltzmann_scores)
        probabilities = [score / total_score for score in boltzmann_scores]
        parents = []
        for _ in range(surviving_N):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents

    def rank_based(self, population, fitness_scores, surviving_N):
        '''Individuals are ranked based on their fitness, and selection probabilities are assigned based on rank rather than absolute fitness.'''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx])
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        total_rank = sum(ranks.values())
        probabilities = [ranks[idx] / total_rank for idx in range(len(population))]
        parents = []
        for _ in range(surviving_N):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents

    def roulette(self, population, fitness_scores, surviving_N):
        ''''Individuals are selected with a probability proportional to their fitness.'''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = []
        for _ in range(surviving_N):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def roulette_linear_scaling(self, population, fitness_scores, surviving_N):
        '''Fitness scores are normalized, and then roulette selection is performed.'''
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        adjusted_scores = [(score - min_fitness) / (max_fitness - min_fitness + 1e-6) for score in fitness_scores]
        return self.roulette(population, adjusted_scores, surviving_N)
    
    def steady_state(self, population, fitness_scores, surviving_N):
        '''
        The k best individuals are selected to be parents, and the worst individuals are replaced by new offspring.
        The remaining individuals remain in the population, unchanged.
        This method only selects the best individuals and does not perform crossover or mutation.
        '''
        return self.truncation(population, fitness_scores, self.steady_state_k, reverse=True)
    
    def sus(self, population, fitness_scores, surviving_N):
        '''
        Similar to roulette wheel selection, but instead of selecting one individual at a time,
        Stochastic Universal Sampling (SUS) uses multiple equally spaced pointers to select individuals simultaneously.
        '''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        step = 1.0 / surviving_N
        start = random.uniform(0, step)
        pointers = [start + i * step for i in range(surviving_N)]
        
        parents = []
        for pointer in pointers:
            for idx, cumulative in enumerate(cumulative_probabilities):
                if pointer <= cumulative:
                    parents.append(population[idx])
                    break
        return parents

    def tournament(self, population, fitness_scores, surviving_N):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected.'''
        parents = []
        for _ in range(surviving_N):
            competitors = random.sample(range(len(population)), k=self.num_competitors)
            winner = max(competitors, key=lambda idx: fitness_scores[idx])
            parents.append(population[winner])
        return parents
    
    def tournament_pop(self, population, fitness_scores, surviving_N):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected and removed from future tournaments.'''
        remaining_population = list(population)
        remaining_fitness_scores = list(fitness_scores)
        parents = []
        for _ in range(surviving_N):
            competitors = random.sample(range(len(remaining_population)), k=self.num_competitors)
            winner_idx = max(competitors, key=lambda idx: remaining_fitness_scores[idx])
            parents.append(remaining_population[winner_idx])
            del remaining_population[winner_idx]
            del remaining_fitness_scores[winner_idx]
        return parents
    
    def tournament_without_replacement(self, population, fitness_scores, surviving_N):
        '''Each individual participates in num_tournaments, with num_competitors individuals and remainder additional participants.'''
        
        total_slots = surviving_N * self.num_competitors
        num_tournaments = total_slots // len(population)
        remainder = total_slots % len(population)
        
        participation_counter = Counter({individual: num_tournaments for individual in population})
        extra_participants = random.sample(population, k=remainder)
        for individual in extra_participants:
            participation_counter[individual] += 1
            
        parents = []
        while len(parents) < surviving_N:
            if len(participation_counter.keys()) == 0:
                break
            competitors = random.sample(list(participation_counter.keys()), k=min(self.num_competitors, len(participation_counter.keys())))
            winner = max(competitors, key=lambda individual: fitness_scores[population.index(individual)])
            parents.append(winner)
            participation_counter[winner] -= 1
            if participation_counter[winner] == 0:
                del participation_counter[winner]

        return parents

    @staticmethod
    def truncation(population, fitness_scores, surviving_N, reverse=False):
        '''
        Only the top individuals are selected for the next generation.
        This method is reused for elitist selection and steady-state selection.
        '''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx], reverse=True)
        if reverse:
            sorted_indices = sorted_indices[::-1]
        parents = [population[idx] for idx in sorted_indices[:surviving_N]]
        return parents