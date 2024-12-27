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

    def PMX(self, parent1, parent2):

        size = len(parent1)
        child1, child2 = [-1]*size, [-1]*size
        
        # Step 1: Select two random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))
        
        # Step 2: Copy the mapping section
        child1[point1:point2+1] = parent2[point1:point2+1]
        child2[point1:point2+1] = parent1[point1:point2+1]
        
        def map_gene(child, parent, point1, point2):
            """Map and resolve conflicts for PMX."""
            for i in range(point1, point2+1):
                if parent[i] not in child:
                    # Resolve conflicts using mapping
                    mapped_value = parent[i]
                    while mapped_value in child[point1:point2+1]:
                        index = parent.index(mapped_value)
                        mapped_value = parent[point1:point2+1][index - point1]
                    if -1 not in child[point1:point2+1]:
                        break
                    child[child.index(-1)] = mapped_value
        
        # Step 3: Resolve conflicts and fill remaining slots
        map_gene(child1, parent1, point1, point2)
        map_gene(child2, parent2, point1, point2)
        
        # Fill remaining empty slots directly from parent1 and parent2
        for i in range(size):
            if child1[i] == -1:
                child1[i] = parent1[i]
            if child2[i] == -1:
                child2[i] = parent2[i]
        
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