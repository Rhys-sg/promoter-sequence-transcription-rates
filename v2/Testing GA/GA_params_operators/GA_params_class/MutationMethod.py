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

    def bit_string(self, infill, generation_idx, generations):
        '''Each nucleotide in the infill has a probability of being mutated.'''
        infill = list(infill)
        for i in range(len(infill)):
            if random.random() < self.mutation_rate:
                infill[i] = random.choice(self.nucleotides)
        return ''.join(infill)
    
    def relative_bit_string(self, infill, generation_idx, generations):
        '''
        Each nucleotide in the infill has a probability of being mutated relative the generation.
        
        When the generation index is 0, the mutation rate is equal to the mutation rate parameter.
        As the generation index increases, the mutation rate decreases by a factor of the relative
        mutation rate alpha parameter

        relative_mutation_rate_alpha ranges from 0 to 1
        '''
        infill = list(infill)
        for i in range(len(infill)):
            modified_mutation_rate = self.relative_mutation_rate_alpha * (generation_idx / generations) + (1 - self.relative_mutation_rate_alpha) * self.mutation_rate
            if random.random() < modified_mutation_rate:
                infill[i] = random.choice(self.nucleotides)
        return ''.join(infill)
    