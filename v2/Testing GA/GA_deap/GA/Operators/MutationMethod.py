import random
import math

class MutationMethod():
    '''
    This class implements various mutation methods for genetic algorithms and stores parameters.
    '''
    def __init__(self, mutation_rate, mutation_rate_start, mutation_rate_end, mutation_rate_degree, generations):
        self.mutation_rate = mutation_rate
        self.mutation_rate_start = mutation_rate_start
        self.mutation_rate_end = mutation_rate_end
        self.mutation_rate_degree = mutation_rate_degree
        self.generation_idx = 0
        self.generations = generations

    @staticmethod
    def mutate(nucleotide):
        '''Randomly change a one-hot encoded nucleotide to another one-hot encoded nucleotide.'''
        nucleotide = [0, 0, 0, 0]
        nucleotide[random.randint(0, 3)] = 1
        return tuple(nucleotide)

    def constant(self, individual, **kwargs):
        '''Each nucleotide in the bit string has a probability of mutating.'''
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = self.mutate(individual[i])
        return (individual,)
    
    def linear(self, individual, generation_idx, **kwargs):
        '''The mutation rate changes linearly over time from the start rate to the end rate.'''
        if generation_idx != self.generation_idx:
            self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * (generation_idx / self.generations)
        return self.constant(individual)
    
    def exponential(self, individual, generation_idx, **kwargs):
        '''The mutation rate changes exponentially over time from the start rate to the end rate.'''
        if generation_idx != self.generation_idx:
            self.generation_idx = generation_idx
            t = self.generation_idx / self.generations
            self.mutation_rate = self.mutation_rate_start + (self.mutation_rate_end - self.mutation_rate_start) * (math.pow(t, self.mutation_rate_degree))
        return self.constant(individual)