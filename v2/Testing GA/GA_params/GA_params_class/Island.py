import random
import numpy as np
import torch

class Island:
    '''
    Each island runs a genetic algorithm to optimize the target expression.
    This class controls the evolution of a population of sequences over multiple generations,
    including selection, crossover, and mutation.

    '''
    def __init__(self, lineage, geneticAlgorithm, idx):
        self.lineage = lineage
        self.geneticAlgorithm = geneticAlgorithm
        self.idx = idx
        self.population = self.initialize_infills()

        self.best_infill = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

    def initialize_infills(self):
        return [
            ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(self.geneticAlgorithm.mask_length)])
            for _ in range(self.geneticAlgorithm.pop_size // self.geneticAlgorithm.islands)
        ]

    def recombination(self, parents):
        parent_chromosomes = [self.split_into_chromosomes(parent) for parent in parents]
        child_chromosomes = []
        for chrom_idx in range(len(parent_chromosomes[0])):
            chrom_slices = [parent[chrom_idx] for parent in parent_chromosomes]
            child_chromosome = self.crossover(chrom_slices)
            child_chromosomes.append(child_chromosome)

        return self.merge_chromosomes(child_chromosomes)
    
    def split_into_chromosomes(self, infill):
        '''Split an infill string into separate chromosomes.'''
        chromosomes = []
        start = 0
        for length in self.geneticAlgorithm.chromosome_lengths:
            chromosomes.append(infill[start:start + length])
            start += length
        return chromosomes

    def merge_chromosomes(self, chromosomes):
        '''Merge chromosomes into a single string.'''
        return ''.join(chromosomes)
    
    @staticmethod
    def crossover(chrom_slices):
        if len(chrom_slices) == 1:
            return chrom_slices[0]
        if len(chrom_slices[0]) == 1:
            return random.choice(chrom_slices)
        
        parent1, parent2 = random.sample(chrom_slices, 2)
        if len(chrom_slices[0]) == 2:
            return parent1[:1] + parent2[1:]
        
        crossover_point = random.randint(1, len(chrom_slices[0]) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    @staticmethod
    def mutate(infill, mutation_rate=0.1):
        infill = list(infill)
        for i in range(len(infill)):
            if random.random() < mutation_rate:
                infill[i] = random.choice(['A', 'C', 'G', 'T'])
        return ''.join(infill)

    def generate_next_population(self):
        fitness_scores, predictions = self.evaluate_population(self.population)
        parents = self.geneticAlgorithm.selection_method(self.population, fitness_scores, self.geneticAlgorithm.surviving_pop)
        next_gen = []

        while len(next_gen) < len(self.population):
            selected_parents = random.sample(parents, self.geneticAlgorithm.num_parents)
            child = self.mutate(self.recombination(selected_parents), self.geneticAlgorithm.base_mutation_rate)
            next_gen.append(child)

        self.update_best(fitness_scores, predictions)
        
        return next_gen[:len(self.population)]
    
    def evaluate_population(self, infills):
        to_evaluate_infills = [infill for infill in infills if infill not in self.geneticAlgorithm.seen_infills]
        if to_evaluate_infills:
            to_evaluate = [
                self.geneticAlgorithm.reconstruct_sequence(infill)
                for infill in to_evaluate_infills
            ]
            predictions = self.predict(to_evaluate)
            fitness_scores = self.calculate_fitness(predictions, to_evaluate)
            for infill, fitness, pred in zip(to_evaluate_infills, fitness_scores, predictions):
                self.geneticAlgorithm.seen_infills[infill] = (fitness, pred)
        fitness_scores = np.array([self.geneticAlgorithm.seen_infills[infill][0] for infill in infills])
        predictions = np.array([self.geneticAlgorithm.seen_infills[infill][1] for infill in infills])
        return fitness_scores, predictions
    
    def predict(self, to_evaluate):
        one_hot_pop = [self.one_hot_sequence(seq.zfill(self.geneticAlgorithm.max_length)) for seq in to_evaluate]
        one_hot_tensor = torch.tensor(np.stack(one_hot_pop), dtype=torch.float32)
        with torch.no_grad():
            predictions = self.geneticAlgorithm.cnn(one_hot_tensor).cpu().numpy().flatten()
        return predictions
    
    def calculate_fitness(self, predictions, to_evaluate):
        '''
        Calculate the fitness of each individual in the population based on:
        - Prediction loss: the negative absolute difference between the predicted and target expression.
        - Hamming loss: the maximum hamming distance between the individual and the best individuals from previous lineages,
                        as a percent of the sequence and scaled by an alpha parameter.
        '''
        prediction_loss = -np.abs(predictions - self.geneticAlgorithm.target_expression)
        if self.geneticAlgorithm.lineage_divergence_alpha == 0:
            return prediction_loss
        hamming_loss = np.array([self.calculate_previous_lineage_hamming(infill) * self.geneticAlgorithm.lineage_divergence_alpha for infill in to_evaluate])
        return prediction_loss + hamming_loss
    
    @staticmethod
    def one_hot_sequence(seq):
        mapping = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25],
            '0': [0, 0, 0, 0]
        }
        return np.array([mapping[nucleotide.upper()] for nucleotide in seq])
    
    def calculate_previous_lineage_hamming(self, infill):
        if self.lineage.lineage_idx != 0:
            return -max(self.calculate_hamm_distance(infill, best_infill)/len(best_infill) for best_infill in self.geneticAlgorithm.best_infills)
        else:
            return 0
    
    @staticmethod
    def calculate_hamm_distance(infill, target_infill):
        return sum([1 for s, t in zip(infill, target_infill) if s != t])
    
    def update_best(self, fitness_scores, predictions):
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_infill = self.population[best_idx]
            self.best_prediction = predictions[best_idx]
            
    def print_progress(self):
        if self.geneticAlgorithm.verbose == 2:
            best_sequence = self.geneticAlgorithm.reconstruct_sequence(
                self.best_infill,
            )
            print(
                f'Lineage {self.lineage.lineage_idx+1} | ' +
                f'Island {self.idx+1} | ' +
                f'Generation {self.lineage.generation_idx+1} | ' +
                f'Best TX rate: {self.best_prediction:.4f} | ' +
                f'Sequence: {best_sequence}'
            )