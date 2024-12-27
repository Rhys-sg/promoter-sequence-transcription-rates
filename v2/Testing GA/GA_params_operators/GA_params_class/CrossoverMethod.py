import random
import numpy as np

class CrossoverMethod():
    '''
    This class implements various crossover methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, k):
        self.k = k
    
    def single_point(self, parent1, parent2):
        '''Single-point crossover selects a random point in the parent sequences and swaps the tails of the sequences.'''
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]
    
    def k_point(self, parent1, parent2):
        '''k-point crossover selects k random points in the parent sequences 
        and alternates between copying segments from each parent.'''
        if self.k < 1:
            return self.single_point(self, parent1, parent2)

        # Generate k unique random crossover points and sort them
        crossover_points = sorted(random.sample(range(1, len(parent1)), self.k))
        
        # Alternate between parents at each crossover point
        child1, child2 = [], []
        last_point = 0
        swap = False
        
        for point in crossover_points:
            if swap:
                child1.extend(parent2[last_point:point])
                child2.extend(parent1[last_point:point])
            else:
                child1.extend(parent1[last_point:point])
                child2.extend(parent2[last_point:point])
            swap = not swap
            last_point = point
        
        # Add the remaining segment after the last crossover point
        if swap:
            child1.extend(parent2[last_point:])
            child2.extend(parent1[last_point:])
        else:
            child1.extend(parent1[last_point:])
            child2.extend(parent2[last_point:])
        
        return ''.join(child1), ''.join(child2)

    
    def uniform(self, parent1, parent2):
        '''Uniform crossover selects genes from each parent with equal probability.'''
        child1 = ''
        child2 = ''
        bool_array = np.random.choice([True, False], size=len(parent1))
        for i in range(len(parent1)):
            if bool_array[i]:
                child1 += parent1[i]
                child2 += parent2[i]
            else:
                child1 += parent2[i]
                child2 += parent1[i]
        return child1, child2