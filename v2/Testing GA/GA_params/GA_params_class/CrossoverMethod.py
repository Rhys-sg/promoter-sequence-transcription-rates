import random
import math
import numpy as np

class CrossoverMethod():
    '''
    This class implements various crossover methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, k, include_boundaries, segments):
        self.k = k
        self.include_boundaries = include_boundaries
        self.segments = segments
    
    def single_point(self, parent1, parent2):
        '''Single-point crossover selects a random point in the parent sequences and swaps the tails of the sequences.'''
        crossover_point = random.randint(1, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]
    
    def k_point(self, parent1, parent2):
        '''K-point crossover selects k random points in the parent sequences and swaps the tails of the sequences.'''
        size = len(parent1)
        points = sorted(random.sample(range(1, size), self.k))
        if self.include_boundaries:
            points = [0] + points + [size]

        child1, child2 = [], []
        swap = False

        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            if swap:
                child1.extend(parent2[start:end])
                child2.extend(parent1[start:end])
            else:
                child1.extend(parent1[start:end])
                child2.extend(parent2[start:end])
            swap = not swap

        return child1, child2
    
    def uniform(self, parent1, parent2):
        '''Uniform crossover selects genes from each parent with equal probability.'''
        child1 = ''
        child2 = ''
        for allele1, allele2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1 += allele1
                child2 += allele2
            else:
                child1 += allele2
                child2 += allele1
        return child1, child2
    
    def PMX(self, parent1, parent2):
        '''Partially-mapped crossover (PMX) selects two random points in the parent sequences and swaps the genes between them.'''
        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size
        point1, point2 = sorted(random.sample(range(size), 2))

        # Copy segment between crossover points
        child1[point1:point2 + 1] = parent2[point1:point2 + 1]
        child2[point1:point2 + 1] = parent1[point1:point2 + 1]

        def fill_remaining(child, parent):
            mapping = {child[i]: parent[i] for i in range(point1, point2 + 1)}
            for i in range(size):
                if child[i] == -1:
                    value = parent[i]
                    while value in mapping:
                        value = mapping[value]
                    child[i] = value

        # Fill remaining values
        fill_remaining(child1, parent1)
        fill_remaining(child2, parent2)

        return child1, child2
    
    def OX(self, parents):
        '''
        Order crossover (OX) copies one (or more) parts of parent to the offspring from the selected cut-points
        and fills the remaining space with values other than the ones included in the copied section.
        '''
        size = len(parents[0])
        child1, child2 = [-1] * size, [-1] * size
        cut_points = sorted(random.sample(range(size), 2 * self.segments))

        # Step 1: Copy selected segments
        for i in range(0, len(cut_points), 2):
            start, end = cut_points[i], cut_points[i + 1]
            child1[start:end] = parents[0][start:end]
            child2[start:end] = parents[1][start:end]

        def fill_child(child, parent):
            idx = 0
            for val in parent:
                while child[idx % size] != -1:
                    idx += 1
                if val not in child:
                    child[idx % size] = val
                    idx += 1

        # Step 2: Fill remaining positions
        fill_child(child1, parents[1])
        fill_child(child2, parents[0])

        return child1, child2
    
    def shuffle(self, parent1, parent2):
        '''Shuffles the values of an individual solution before the crossover and unshuffles them after crossover operation is performed'''
        size = len(parent1)
        indices = list(range(size))
        shuffled_indices = indices[:]
        random.shuffle(shuffled_indices)

        # Shuffle parents
        shuffled_parent1 = [parent1[i] for i in shuffled_indices]
        shuffled_parent2 = [parent2[i] for i in shuffled_indices]

        # Perform crossover (single-point in this case)
        crossover_point = random.randint(1, size - 1)
        child1 = shuffled_parent1[:crossover_point] + shuffled_parent2[crossover_point:]
        child2 = shuffled_parent2[:crossover_point] + shuffled_parent1[crossover_point:]

        # Unshuffle children
        unshuffled_child1 = [None] * size
        unshuffled_child2 = [None] * size
        for i, index in enumerate(shuffled_indices):
            unshuffled_child1[index] = child1[i]
            unshuffled_child2[index] = child2[i]

        return unshuffled_child1, unshuffled_child2