import random
import math

class MutationMethod():
    '''
    This class implements various mutation methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, mutation_rate, generations, relative_mutation_rate_alpha):
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.relative_mutation_rate_alpha = relative_mutation_rate_alpha
        self.nucleotides = ['A', 'C', 'G', 'T']

    def bit_string(self, infill, generation_idx):
        '''Each nucleotide in the infill has a probability of being mutated.'''
        for i in range(len(infill)):
            if random.random() < self.mutation_rate:
                infill[i] = random.choice(self.nucleotides)
        return ''.join(infill)
    
    def relative_bit_string(self, infill, generation_idx):
        '''Each nucleotide in the infill has a probability of being mutated relative the generation.'''
        for i in range(len(infill)):
            if random.random() < self.mutation_rate * (self.mutation_rate / (generation_idx + 1)):
                infill[i] = random.choice(self.nucleotides)
        return ''.join(infill)
    