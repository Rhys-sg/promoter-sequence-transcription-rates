import random
import numpy as np
from itertools import combinations

class ParentChoiceMethod():
    '''
    This class implements various methods for choosing parents for crossover.
    '''
    def __init__(self, covariance, generational_covariance_alpha):
        self.covariance = covariance
        self.generational_covariance_alpha = generational_covariance_alpha

    def by_order(self, parents, generation_idx, generations):
        ''' Pair parents based on the order of the parents list. '''
        paired_parents = []
        for i in range(0, len(parents) - 1, 2):
            paired_parents.append((parents[i], parents[i + 1]))
        return paired_parents
    
    def without_replacement(self, parents, generation_idx, generations):
        ''' Pair parents randomly, without replacement. '''
        random.shuffle(parents)
        paired_parents = []
        for i in range(0, len(parents) - 1, 2):
            paired_parents.append((parents[i], parents[i + 1]))
        return paired_parents
    
    def with_replacement(self, parents, generation_idx, generations):
        ''' Pair parents randomly, with replacement. '''
        paired_parents = []
        while len(paired_parents) < len(parents) // 2:
            paired_parents.append(random.sample(parents, 2))
        return paired_parents
    
    def by_covariance(self, parents, generation_idx, generations):
        ''' Pair parents based on the covariance of their hamming distance. '''
        ordered_distances = self.ordered_by_hamming_distance(parents)
        paired_parents = []
        for parent1_idx in range(len(parents) // 2):
            parent1 = parents[parent1_idx]   
            parent2_idx = self.covariance_index(self.modified_covariance(generation_idx, generations), len(parents))
            parent2 = ordered_distances[parent1][parent2_idx]
            paired_parents.append((parent1, parent2))
        
        return paired_parents

    def ordered_by_hamming_distance(self, population):
        ordered_distances = {}
        for i, ind in enumerate(population):
            distances = [(other, self.hamming_distance(ind, other)) for j, other in enumerate(population) if i != j]
            distances.sort(key=lambda x: x[1])
            ordered_distances[ind] = [other for other, dist in distances]
        return ordered_distances

    @staticmethod
    def hamming_distance(parent1, parent2):
        ''' Calculate the Hamming distance between two sequences. '''
        return sum([1 for i in range(len(parent1)) if parent1[i] != parent2[i]]) / len(parent1)
    
    def modified_covariance(self, generation_idx, generations):
        '''
        Calculate the modified covariance based on generational_covariance_alpha and the current generation.
        
        If generational_covariance_alpha is None, return the covariance parameter.
        modified covariance is calculated by linearly interpolating between the covariance parameter
        and the generational covariance alpha parameter based on the current generation.

        When the generation index is 0, the modified covariance is equal to the covariance parameter.
        As the generation index increases, the modified covariance decreases by a factor of the generational
        covariance alpha parameter.
        '''
        if self.generational_covariance_alpha != None:
            return self.generational_covariance_alpha * (generation_idx / generations) + (1 - self.generational_covariance_alpha) * self.covariance
        
        return self.covariance

    @staticmethod
    def covariance_index(covariance, length, k=5):
        covariance = np.clip((covariance + 1) / 2, 0, 1)
        alpha = 1 + k * covariance
        beta = 1 + k * (1 - covariance)
        return int(np.random.beta(alpha, beta) * length) - 1
    