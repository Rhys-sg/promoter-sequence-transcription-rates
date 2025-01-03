import random
import math
from collections import Counter
from deap import tools  # type: ignore

class SelectionMethod():
    '''
    This class implements various selection methods for genetic algorithms and stores selection parameters.
    It contains methods references to the ones in the DEAP library, as well as custom methods.

    Not applicable DEAP methods:
    - selWorst
    - selDoubleTournament

    Multi-objective DEAP methods (not umplemented yet):
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
        return self.selRandom(*args, **kwargs)
    
    def selBest(self, *args, **kwargs):
        return self.selBest(*args, **kwargs)
    
    def selTournament(self, *args, **kwargs):
        return self.selTournament(*args, **kwargs)
    
    def selRoulette(self, *args, **kwargs):
        return self.selRoulette(*args, **kwargs)
    
    def selStochasticUniversalSampling(self, *args, **kwargs):
        return self.selStochasticUniversalSampling(*args, **kwargs)
    
    '''
    The following methods are custom implementations:
    '''

    def selBoltzmann(self, individuals, k):
        '''
        Based on simulated annealing, this method adjusts selection probabilities dynamically over time,
        favoring exploration in early generations and exploitation in later generations.
        '''
        fitness_scores = [ind.fitness.values[0] for ind in individuals]
        boltzmann_scores = [math.exp(score / self.boltzmann_temp) for score in fitness_scores]
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
        normalized_scores = [(score - min_fitness) / (max_fitness - min_fitness) for score in fitness_scores]
        for ind, norm_score  in zip(individuals, normalized_scores):
            ind.fitness.normalized = (norm_score,)
        selected = tools.selRoulette(individuals, k, fit_attr='fitness.normalized')
        for ind in individuals:
            if hasattr(ind.fitness, 'normalized'):
                del ind.fitness.normalized
        return selected
    
    def selTournamentWithoutReplacement(self, individuals, k):
        '''Each individual participates in num_tournaments, with tournament_size individuals and remainder additional participants.'''

        total_slots = k * self.tournsize
        num_tournaments = total_slots // len(individuals)
        remainder = total_slots % len(individuals)
        
        participation_counter = Counter({individual: num_tournaments for individual in individuals})
        extra_participants = random.sample(individuals, k=remainder)
        for individual in extra_participants:
            participation_counter[individual] += 1
            
        chosen = []
        while len(chosen) < k:
            if len(participation_counter.keys()) == 0:
                break
            aspirants = tools.selRandom(individuals, self.tournsize)
            winner = max(aspirants, key=lambda ind: ind.fitness.values[0])
            chosen.append(winner)
            participation_counter[winner] -= 1
            if participation_counter[winner] == 0:
                del participation_counter[winner]

        return chosen
    
    def get_all_methods():
        return [method for method in dir(SelectionMethod) if method.startswith('sel')]