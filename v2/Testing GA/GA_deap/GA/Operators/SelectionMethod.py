import random
import math
from collections import Counter
from deap import tools  # type: ignore

class SelectionMethod():
    '''
    This class implements various selection methods for genetic algorithms and stores selection parameters.
    It contains method references to the DEAP library, as well as custom methods.

    Does not include:
    - selWorst
    - selDoubleTournament

    Or Multi-objective DEAP methods:
    - selLexicase
    - selEpsilonLexicase
    - selAutomaticEpsilonLexicase

    '''
    def __init__(self, boltzmann_temperature, tournsize):
        self.boltzmann_temperature = boltzmann_temperature
        self.tournsize = tournsize
    
    '''
    The following methods have already been implemented in the DEAP library:
    '''
    def selRandom(self, *args, **kwargs):
        return tools.selRandom(*args, **kwargs)
    
    def selBest(self, *args, **kwargs):
        return tools.selBest(*args, **kwargs)
    
    def selTournament(self, *args, **kwargs):
        return tools.selTournament(tournsize=self.tournsize, *args, **kwargs)
    
    def selRoulette(self, *args, **kwargs):
        return tools.selRoulette(*args, **kwargs)
    
    def selStochasticUniversalSampling(self, *args, **kwargs):
        return tools.selStochasticUniversalSampling(*args, **kwargs)
    
    '''
    The following methods are custom implementations:
    '''

    def selBoltzmann(self, individuals, k):
        '''
        Based on simulated annealing, this method adjusts selection probabilities dynamically over time,
        favoring exploration in early generations and exploitation in later generations.
        '''
        fitness_scores = [ind.fitness.values[0] for ind in individuals]
        boltzmann_scores = [math.exp(score / self.boltzmann_temperature) for score in fitness_scores]
        total_score = sum(boltzmann_scores)
        probabilities = [score / total_score for score in boltzmann_scores]
        
        parents = []
        for _ in range(k):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(individuals[idx])
                    break
        return parents
    

    def selNormRoulette(self, individuals, k, **kwargs):
        '''Select the k best individuals according to their normalized fitness.'''
        fitness_scores = [ind.fitness.values[0] for ind in individuals]
        min_fitness = min(fitness_scores)
        max_fitness = max(fitness_scores)
        if max_fitness - min_fitness == 0:
            normalized_scores = [1.0 for _ in fitness_scores]
        else:
            normalized_scores = [(score - min_fitness) / (max_fitness - min_fitness) for score in fitness_scores]

        s_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)
        sum_fits = sum(getattr(ind, 'fitness').values[0] for ind in s_inds)
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += getattr(ind, 'fitness').values[0]
                if sum_ > u:
                    chosen.append(ind)
                    break

        return chosen

    
    def selTournamentWithoutReplacement(self, individuals, k):
        '''Each individual participates in num_tournaments, with tournament_size individuals and remainder additional participants.'''

        # Assign unique indices to individuals
        individual_indices = list(range(len(individuals)))

        total_slots = k * self.tournsize
        num_tournaments = total_slots // len(individuals)
        remainder = total_slots % len(individuals)
        
        # Use indices for the Counter
        participation_counter = Counter({idx: num_tournaments for idx in individual_indices})
        extra_participants = random.sample(individual_indices, k=remainder)
        for idx in extra_participants:
            participation_counter[idx] += 1
            
        chosen = []
        while len(chosen) < k:
            if len(participation_counter.keys()) == 0:
                break
            aspirant_indices = random.sample(individual_indices, self.tournsize)
            aspirants = [individuals[idx] for idx in aspirant_indices]
            winner = max(aspirants, key=lambda ind: ind.fitness.values[0])
            winner_idx = individuals.index(winner)
            chosen.append(winner)
            participation_counter[winner_idx] -= 1
            if participation_counter[winner_idx] == 0:
                del participation_counter[winner_idx]

        return chosen

    
    def get_all_methods():
        return [method for method in dir(SelectionMethod) if method.startswith('sel')]