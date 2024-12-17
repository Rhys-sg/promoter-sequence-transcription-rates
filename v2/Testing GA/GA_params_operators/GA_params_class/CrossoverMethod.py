import random

class CrossoverMethod():
    '''
    This class implements various crossover methods for genetic algorithms and allows multi-parent support.
    If len(parents) = 1, it returns the same parent.
    '''
    def __init__(self, k, include_boundaries, segments):
        self.k = k
        self.include_boundaries = include_boundaries
        self.segments = segments

    def single_point(self, parents):
        '''Single-point crossover with multiple parents. It swaps tails at a random point for each parent pair sequentially.'''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        crossover_point = random.randint(1, size - 1)
        children = []
        
        for i in range(len(parents)):
            next_parent = parents[(i + 1) % len(parents)]
            child = parents[i][:crossover_point] + next_parent[crossover_point:]
            children.append(child)
        return children

    def k_point(self, parents):
        '''K-point crossover with multiple parents. It swaps segments at k random points for each pair. '''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        points = sorted(random.sample(range(1, size), self.k))
        if self.include_boundaries:
            points = [0] + points + [size]
        
        children = []
        for i in range(len(parents)):
            child = []
            swap = False
            parent_a, parent_b = parents[i], parents[(i + 1) % len(parents)]
            
            for j in range(len(points) - 1):
                start, end = points[j], points[j + 1]
                if swap:
                    child.extend(parent_b[start:end])
                else:
                    child.extend(parent_a[start:end])
                swap = not swap
            children.append(child)
        return children
    
    def uniform(self, parents):
        '''Uniform crossover with multiple parents. Each gene is taken from a random parent.'''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        children = []
        
        for _ in range(len(parents)):
            child = ''
            for i in range(size):
                chosen_parent = random.choice(parents)
                child += chosen_parent[i]
            children.append(child)
        return children

    def PMX(self, parents):
        '''Partially-mapped crossover (PMX) for multiple parents. Crossover is applied pairwise. '''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        children = []
        
        for i in range(len(parents)):
            parent_a = parents[i]
            parent_b = parents[(i + 1) % len(parents)]
            child = [-1] * size
            point1, point2 = sorted(random.sample(range(size), 2))
            
            # Copy segment
            child[point1:point2 + 1] = parent_b[point1:point2 + 1]
            mapping = {child[j]: parent_a[j] for j in range(point1, point2 + 1)}
            
            # Fill remaining genes
            for j in range(size):
                if child[j] == -1:
                    value = parent_a[j]
                    while value in mapping:
                        value = mapping[value]
                    child[j] = value
            children.append(child)
        return children

    def OX(self, parents):
        '''Order crossover (OX) with multiple parents. Each child takes ordered segments from parents. '''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        cut_points = sorted(random.sample(range(size), 2 * self.segments))
        children = []

        for i in range(len(parents)):
            child = [-1] * size
            parent_a = parents[i]
            parent_b = parents[(i + 1) % len(parents)]
            
            # Step 1: Copy segments
            for j in range(0, len(cut_points), 2):
                start, end = cut_points[j], cut_points[j + 1]
                child[start:end] = parent_a[start:end]
            
            # Step 2: Fill remaining positions
            idx = 0
            for val in parent_b:
                while child[idx % size] != -1:
                    idx += 1
                if val not in child:
                    child[idx % size] = val
                    idx += 1
            children.append(child)
        return children
    
    def shuffle(self, parents):
        '''Shuffle crossover with multiple parents. It shuffles, crossovers, and unshuffles the sequences.'''
        if len(parents) == 1:
            return parents
        
        size = len(parents[0])
        indices = list(range(size))
        shuffled_indices = indices[:]
        random.shuffle(shuffled_indices)
        
        children = []
        for i in range(len(parents)):
            next_parent = parents[(i + 1) % len(parents)]
            shuffled_a = [parents[i][j] for j in shuffled_indices]
            shuffled_b = [next_parent[j] for j in shuffled_indices]
            
            # Perform single-point crossover
            crossover_point = random.randint(1, size - 1)
            child = shuffled_a[:crossover_point] + shuffled_b[crossover_point:]
            
            # Unshuffle
            unshuffled_child = [None] * size
            for idx, original_idx in enumerate(shuffled_indices):
                unshuffled_child[original_idx] = child[idx]
            children.append(unshuffled_child)
        return children
