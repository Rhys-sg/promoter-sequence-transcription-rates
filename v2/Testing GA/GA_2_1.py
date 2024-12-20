import torch
import numpy as np
import random
import re
from keras.models import load_model  # type: ignore


class GeneticAlgorithm:
    """
    This class performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    This version splits the masked sequence into multiple chromosomes, each filled independently using crossover and mutation operations.

    """

    def __init__(self, cnn_model_path, masked_sequence, target_expression, max_length=150, pop_size=20, generations=100, base_mutation_rate=0.1, precision=0.01, chromosomes=1, print_progress=True):
        self.device = self.get_device()
        self.cnn = load_model(cnn_model_path)
        self.masked_sequence = masked_sequence
        self.target_expression = target_expression
        self.max_length = max_length
        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = base_mutation_rate
        self.precision = precision
        self.chromosomes = chromosomes
        self.print_progress = print_progress
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.chromosome_lengths = self._split_chromosome_lengths(self.mask_length, chromosomes)
        self.best_infill = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

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

    def _split_chromosome_lengths(self, total_length, chromosomes):
        """Split the mask length into chromosome lengths."""
        base_length = total_length // chromosomes
        lengths = [base_length] * chromosomes
        for i in range(total_length % chromosomes):
            lengths[i] += 1
        return lengths

    def _split_into_chromosomes(self, infill):
        """Split an infill string into separate chromosomes."""
        chromosomes = []
        start = 0
        for length in self.chromosome_lengths:
            chromosomes.append(infill[start:start + length])
            start += length
        return chromosomes

    def _merge_chromosomes(self, chromosomes):
        """Merge chromosomes into a single string."""
        return ''.join(chromosomes)

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
    def select_parents(population, fitness_scores, num_parents):
        parents = []
        for _ in range(num_parents):
            competitors = random.sample(range(len(population)), k=5)
            winner = max(competitors, key=lambda idx: fitness_scores[idx])
            parents.append(population[winner])
        return parents

    def crossover(self, parent1, parent2):
        """Perform crossover on two infills by splitting them into chromosomes."""
        chromosomes1 = self._split_into_chromosomes(parent1)
        chromosomes2 = self._split_into_chromosomes(parent2)
        child1_chromosomes, child2_chromosomes = [], []

        for c1, c2 in zip(chromosomes1, chromosomes2):
            child1, child2 = list(c1), list(c2)
            if len(c1) > 1:
                crossover_point = random.randint(1, len(c1) - 1)
                child1[crossover_point:], child2[crossover_point:] = c2[crossover_point:], c1[crossover_point:]
            child1_chromosomes.append(''.join(child1))
            child2_chromosomes.append(''.join(child2))

        return self._merge_chromosomes(child1_chromosomes), self._merge_chromosomes(child2_chromosomes)

    @staticmethod
    def mutate(infill, mutation_rate=0.1):
        infill = list(infill)
        for i in range(len(infill)):
            if random.random() < mutation_rate:
                infill[i] = random.choice(['A', 'C', 'G', 'T'])
        return ''.join(infill)

    def run(self):
        infills = self.initialize_infills(self.mask_length, self.pop_size)

        for gen in range(self.generations):
            fitness_scores, predictions = self.evaluate_population(infills)

            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_infill = infills[best_idx]
                self.best_prediction = predictions[best_idx]

            if self.print_progress:
                best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_infill, self.mask_indices)
                print(f"Generation {gen+1} | Best TX rate: {self.best_prediction:.4f} | Target TX rate: {self.target_expression} | Sequence: {best_sequence}")

            if abs(self.best_prediction - self.target_expression) < self.precision:
                if self.print_progress:
                    print("Early stopping as target TX rate is achieved.")
                break

            parents = self.select_parents(infills, fitness_scores, self.pop_size // 2)
            next_gen = []
            while len(next_gen) < self.pop_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_gen.append(self.mutate(child1, self.base_mutation_rate))
                next_gen.append(self.mutate(child2, self.base_mutation_rate))
            infills = next_gen

        best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_infill, self.mask_indices)
        return best_sequence, self.best_prediction


if __name__ == '__main__':
    cnn_model_path = 'v2/Models/CNN_6_1_2.keras'
    masked_sequence = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'
    target_expression = 1

    ga = GeneticAlgorithm(
        cnn_model_path=cnn_model_path,
        masked_sequence=masked_sequence,
        target_expression=target_expression,
        pop_size=100,
        generations=100,
        base_mutation_rate=0.1,
        precision=0.001,
        chromosomes=3,
        print_progress=True
    )
    best_sequence, best_prediction = ga.run()
    print("\nBest infilled sequence:", best_sequence)
    print("Predicted transcription rate:", best_prediction)
