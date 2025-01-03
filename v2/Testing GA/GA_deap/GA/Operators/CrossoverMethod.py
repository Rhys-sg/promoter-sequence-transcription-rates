import random
import numpy as np
from deap import tools # type: ignore

class CrossoverMethod():
    '''
    This class implements various crossover methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, k, uniformProb):
        self.k = k
        self.uniformProb = uniformProb
    
    def cxOnePoint(self, parent1, parent2):
        return tools.cxOnePoint(parent1, parent2)
    
    def cxTwoPoint(self, parent1, parent2):
        return tools.cxTwoPoint(parent1, parent2)
    
    def cxUniform(self, parent1, parent2):
        return tools.cxUniform(parent1, parent2, self.uniformProb)
    
    def cxKPoint(self, parent1, parent2):
        '''k-point crossover selects k random points in the parent sequences and alternates between copying segments from each parent.'''
        if self.k < 1:
            return self.single_point(self, parent1, parent2)

        crossover_points = sorted(random.sample(range(1, len(parent1)), self.k))
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
        
        if swap:
            child1.extend(parent2[last_point:])
            child2.extend(parent1[last_point:])
        else:
            child1.extend(parent1[last_point:])
            child2.extend(parent2[last_point:])
        
        return tuple(child1), tuple(child2)
    
    def cxUniform(self, parent1, parent2):
        '''Uniform crossover selects genes from each parent with equal probability.'''
        child1 = []
        child2 = []
        bool_array = np.random.choice([True, False], size=len(parent1))
        for i in range(len(parent1)):
            if bool_array[i]:
                child1 += parent1[i]
                child2 += parent2[i]
            else:
                child1 += parent2[i]
                child2 += parent1[i]
        return tuple(child1), tuple(child2)
    
    def get_all_methods():
        return [method for method in dir(CrossoverMethod) if method.startswith('cx')]