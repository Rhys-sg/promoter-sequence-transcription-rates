import random
import numpy as np
import torch

class Island:
    '''
    Each island runs a genetic algorithm to optimize the target expression.
    This class controls the evolution of a population of sequences over multiple generations,
    including selection, crossover, and mutation.

    '''
    def __init__(self, lineage, geneticAlgorithm, island_idx):
        self.lineage = lineage
        self.geneticAlgorithm = geneticAlgorithm
        self.island_idx = island_idx
        self.population = self.initialize_infills()

        # operators
        self.crossover_method = geneticAlgorithm.crossover_method
        self.mutation_method = geneticAlgorithm.mutation_method
        self.selection_method = geneticAlgorithm.selection_method
        self.parent_choice_method = geneticAlgorithm.parent_choice_method

        self.best_infill = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

        self.infill_history = []
        self.fitness_history = []
        self.prediction_history = []

    def initialize_infills(self):
        return [
            ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(self.geneticAlgorithm.mask_length)])
            for _ in range(self.geneticAlgorithm.pop_size // self.geneticAlgorithm.islands)
        ]

    def generate_next_population(self):
        fitness_scores, predictions = self.evaluate_population(self.population)
        self.record_history(fitness_scores, predictions)
        self.update_best(fitness_scores, predictions)
        
        parents = self.selection_method(self.population, fitness_scores, self.geneticAlgorithm.island_pop // 2)
        paired_parents = self.parent_choice_method(parents, self.lineage.generation_idx, self.geneticAlgorithm.generations)
        next_gen = []

        for parent1, parent2 in paired_parents:
            parent1, parent2 = random.sample(parents, 2)
            children = self.crossover_method(parent1, parent2)
            children = [self.mutation_method(child, self.lineage.generation_idx, self.geneticAlgorithm.generations) for child in children]
            next_gen.extend(children)
        
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
        lineage_divergence_loss = self.calculate_previous_lineage_hamming(to_evaluate, self.geneticAlgorithm.lineage_divergence_alpha)
        diversity_loss = self.calculate_diversity_hamming(to_evaluate, self.geneticAlgorithm.diversity_alpha)
        return prediction_loss + lineage_divergence_loss + diversity_loss
    
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
    
    def calculate_previous_lineage_hamming(self, to_evaluate, alpha):
        if self.lineage.lineage_idx == 0 or alpha == 0:
            return np.zeros(len(to_evaluate))
        hamming_distances = []
        for infill in to_evaluate:
            hamming_distances.append(-max([self.calculate_hamm_distance(infill, best_infill)/len(infill) for best_infill in self.geneticAlgorithm.best_infills]) * alpha)
        return np.array(hamming_distances)
        
    def calculate_diversity_hamming(self, to_evaluate, alpha):
        if alpha == 0:
            return np.zeros(len(to_evaluate))
        distances = []
        for i, infill in enumerate(to_evaluate):
            distances.append(-sum(self.calculate_hamm_distance(infill, other_infill)/len(infill) for other_infill in to_evaluate[:i:]) / (len(to_evaluate)-1) * alpha)
        return np.array(distances) 

    @staticmethod
    def calculate_hamm_distance(infill, target_infill):
        return sum([1 for s, t in zip(infill, target_infill) if s != t])
    
    def record_history(self, fitness_scores, predictions):
        self.infill_history.append(self.population)
        self.fitness_history.append(fitness_scores)
        self.prediction_history.append(predictions)
    
    def update_best(self, fitness_scores, predictions):
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_infill = self.population[best_idx]
            self.best_fitness = fitness_scores[best_idx]
            self.best_prediction = predictions[best_idx]
            
    def print_progress(self):
        if self.geneticAlgorithm.verbose > 1:
            best_sequence = self.geneticAlgorithm.reconstruct_sequence(
                self.best_infill,
            )
            print(
                f'Lineage {self.lineage.lineage_idx+1} | ' +
                f'Island {self.island_idx+1} | ' +
                f'Generation {self.lineage.generation_idx+1} | ' +
                f'Best TX rate: {self.best_prediction:.4f} | ' +
                f'Sequence: {best_sequence}'
            )
    
    def get_infill_history(self):
        return self.infill_history
    
    def get_fitness_history(self):
        return self.fitness_history
    
    def get_prediction_history(self):
        return self.prediction_history