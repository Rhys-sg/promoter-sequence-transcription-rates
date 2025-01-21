import random
import math
import numpy as np

class MutationMethod:
    '''
    This class implements various mutation methods for genetic algorithms and stores parameters.
    '''
    def __init__(self, mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, inverse_entropy, generations):
        self.mutation_rate = mutation_rate
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_end = mutation_rate_end
        self.mutation_rate_degree = mutation_rate_degree
        self.inverse_entropy = inverse_entropy
        self.generation_idx = 0
        self.generations = generations

    @staticmethod
    def _mutate(nucleotide):
        '''Randomly change a one-hot encoded nucleotide to another one-hot encoded nucleotide.'''
        nucleotide = [0, 0, 0, 0]
        nucleotide[random.randint(0, 3)] = 1
        return tuple(nucleotide)

    def mutConstant(self, individual, **kwargs):
        '''Each nucleotide in the bit string has a probability of mutating.'''
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = self._mutate(individual[i])
        return (individual,)
    
    def mutLinear(self, individual, **kwargs):
        '''The mutation rate changes linearly over time from the start rate to the end rate.'''
        return self.mutConstant(individual)
    
    def mutExponential(self, individual, **kwargs):
        '''The mutation rate changes exponentially over time from the start rate to the end rate.'''
        return self.mutConstant(individual)
    
    def mutEntropy(self, individual, population, **kwargs):
        '''
        Adjust mutation rate based on population entropy, mutation start rate, and end rate.
        '''
        return self.mutConstant(individual)
    
    def update_rate(self, population, generation_idx):
        """Adjust the mutation rate based on the generation or population."""
        self.generation_idx = generation_idx
        t = self.generation_idx / self.generations
        if hasattr(self, 'mutLinear'):
            self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * t
        elif hasattr(self, 'mutExponential'):
            self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * (t ** self.mutation_rate_degree)
        elif hasattr(self, 'mutEntropy'):
            entropy_effect = self._calculate_entropy(population) / 2
            if self.inverse_entropy:
                entropy_effect = 1 - entropy_effect
            self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * entropy_effect

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
