import torch
import numpy as np
import random
import re
from keras.models import load_model  # type: ignore
import math


class GeneticAlgorithm:
    """
    This class performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    Includes multiple parents, multiple competitors for tournament selection, and seperate lineages.

    """

    def __init__(self, cnn_model_path, masked_sequence, target_expression, pop_size=100, generations=100,
            base_mutation_rate=0.05, precision=0.001, num_parents=20, num_competitors=5, survival_rate=0.5, seed=None, print_progress=True):
        self.device = self.get_device()
        self.cnn = load_model(cnn_model_path)
        self.masked_sequence = masked_sequence
        self.target_expression = target_expression
        self.max_length = 150 # CNN cannot handle sequences longer than 150
        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = base_mutation_rate
        self.precision = precision
        self.num_parents = min(num_parents, pop_size//2)
        self.num_competitors = min(num_competitors, pop_size)
        self.survival_rate = survival_rate
        self.print_progress = print_progress
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.best_infill = None
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    @staticmethod
    def find_masked_regions(sequence):
        return [(m.start(), m.end()) for m in re.finditer('N+', sequence)]

    @staticmethod
    def initialize_infills(mask_length, pop_size):
        return [
            ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(mask_length)])
            for _ in range(pop_size)
        ]

    @staticmethod
    def reconstruct_sequence(base_sequence, infill, mask_indices):
        sequence = list(base_sequence)
        for idx, char in zip(mask_indices, infill):
            sequence[idx] = char
        return ''.join(sequence)

    def evaluate_population(self, infills):
        full_population = [
            self.reconstruct_sequence(self.masked_sequence, infill, self.mask_indices)
            for infill in infills
        ]
        one_hot_pop = [self.one_hot_sequence(seq.zfill(self.max_length)) for seq in full_population]
        one_hot_tensor = torch.tensor(np.stack(one_hot_pop), dtype=torch.float32)
        with torch.no_grad():
            predictions = self.cnn(one_hot_tensor).cpu().numpy().flatten()
        fitness_scores = -np.abs(predictions - self.target_expression)
        return fitness_scores, predictions
    
    @staticmethod
    def select_parents(population, fitness_scores, surviving_pop, temperature):
        """
        Selects parents using Boltzmann selection. 
        Inspired by simulated annealing, selection is based on a temperature-controlled
        probability that determines the fitness influence over selection.
        Initially, less fit individuals have a higher chance, but as the "temperature" decreases,
        selection favors fitter individuals.
        """
        boltzmann_scores = [math.exp(score / temperature) for score in fitness_scores]
        total_score = sum(boltzmann_scores)
        probabilities = [score / total_score for score in boltzmann_scores]
        parents = []
        for _ in range(surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents

    @staticmethod
    def crossover(parent1, parent2):
        child1, child2 = list(parent1), list(parent2)
        if len(parent1) > 1:  # Ensure there's space for crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child1[crossover_point:], child2[crossover_point:] = parent2[crossover_point:], parent1[crossover_point:]
        return ''.join(child1), ''.join(child2)
    
    # TODO: Implement multiple-parent crossover. Does it do anything without multiple chromosomes?
    # @staticmethod
    # def crossover(parents):
    #     children = []
    #     if len(parents[0]) > 1:  # Ensure there's space for crossover
    #         crossover_point = random.randint(1, len(parents[0]) - 1)

    #         chrom_slices = [parent[chrom_idx] for parent in parent_chromosomes]
    #         child = ''.join(random.choice(chrom_slices)[i] for i in range(len(chrom_slices[0])))
    #         children.append(child)
    #     return children

    @staticmethod
    def mutate(infill, mutation_rate=0.1):
        infill = list(infill)
        for i in range(len(infill)):
            if random.random() < mutation_rate:
                infill[i] = random.choice(['A', 'C', 'G', 'T'])
        return ''.join(infill)

    def run_lineage(self, lineage):
        
        infills = self.initialize_infills(self.mask_length, self.pop_size)
        
        self.best_fitness = -float('inf')
        self.best_infill = None
        self.best_prediction = 0

        for gen in range(self.generations):
            fitness_scores, predictions = self.evaluate_population(infills)

            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_infill = infills[best_idx]
                self.best_prediction = predictions[best_idx]

            if self.print_progress:
                best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_infill, self.mask_indices)
                print(f"Lineage {lineage+1} | Generation {gen+1} | Best TX rate: {self.best_prediction:.4f} | Target TX rate: {self.target_expression} | Sequence: {best_sequence}")

            if abs(self.best_prediction - self.target_expression) < self.precision:
                if self.print_progress:
                    print("Early stopping as target TX rate is achieved.")
                break

            parents = self.select_parents(infills, fitness_scores, int(self.pop_size * self.survival_rate), self.num_competitors)
            next_gen = []
            while len(next_gen) < self.pop_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_gen.append(self.mutate(child1, self.base_mutation_rate))
                next_gen.append(self.mutate(child2, self.base_mutation_rate))
            infills = next_gen

        best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_infill, self.mask_indices)
        return best_sequence, self.best_prediction
    
    def run(self, lineages=1):
        best_sequences = []
        best_predictions = []
        for lineage in range(lineages):
            best_sequence, best_prediction = self.run_lineage(lineage)
            best_sequences.append(best_sequence)
            best_predictions.append(best_prediction)
        return best_sequences, best_predictions

if __name__ == '__main__':
    cnn_model_path = 'v2/Models/CNN_6_1_2.keras'
    masked_sequence = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'
    target_expression = 0.9

    ga = GeneticAlgorithm(
        cnn_model_path=cnn_model_path,
        masked_sequence=masked_sequence,
        target_expression=0.9,
        print_progress=True
    )
    best_sequences, best_predictions = ga.run(5)
    print("\nBest infilled sequence:", best_sequences)
    print("Predicted transcription rate:", best_predictions)
