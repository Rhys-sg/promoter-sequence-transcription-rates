import random
import math
import numpy as np

class MutationMethod():
    '''
    This class implements various mutation methods for genetic algorithms and stores parameters.

    For each mutation method, the method must take in the individual to mutate and return the mutated individual.
    All but mutConstant adjust the mutation rate over time. This is done only once per generation, so the generation index must be passed in.
    '''
    def __init__(self, mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations):
        self.mutation_rate = mutation_rate
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_end = mutation_rate_end
        self.mutation_rate_degree = mutation_rate_degree
        self.generation_idx = 0
        self.generations = generations

    def mutConstant(self, **kwargs):
        '''The mutation rate remains constant over time.'''
        return self.mutation_rate
    
    def mutLinear(self, generation_idx, **kwargs):
        '''The mutation rate changes linearly over time from the start rate to the end rate.'''
        self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * (generation_idx / self.generations)
        return self.mutation_rate
    
    def mutExponential(self, generation_idx, **kwargs):
        '''The mutation rate changes exponentially over time from the start rate to the end rate.'''
        self.generation_idx = generation_idx
        t = self.generation_idx / self.generations
        self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * (math.pow(t, self.mutation_rate_degree))
        return self.mutation_rate
    
    def mutEntropy(self, population, **kwargs):
        '''The mutation rate changes based on the entropy of the population.'''
        entropy_effect = 1 - self._calculate_entropy(population) / 2
        self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * entropy_effect
        return self.mutation_rate

    @staticmethod
    def _calculate_entropy(population):
        '''
        Calculate the average entropy of the population based on the entropy of each index in the population.
        Returns value between 0 and 2. 0 means all sequences are the same, 2 means all sequences are different.
        '''
        entropies = []
        for index in range(len(population[0])):
            frequency = {(0, 0, 0, 1): 0, (0, 0, 1, 0): 0, (0, 1, 0, 0): 0, (1, 0, 0, 0): 0, (0, 0, 0, 0): 0}
            for sequence in population:
                frequency[sequence[index]] += 1
            total_count = sum(frequency.values())
            probabilities = [freq / total_count for freq in frequency.values() if freq > 0]
            entropy = -sum(p * np.log2(p) for p in probabilities)
            entropies.append(entropy)
        return sum(entropies) / len(entropies)
    
    @staticmethod
    def get_all_methods():
        return [method for method in dir(MutationMethod) if method.startswith('mut')]
