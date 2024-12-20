import torch
import numpy as np
import random
import re
import math
from keras.models import load_model  # type: ignore


class GeneticAlgorithm:
    """
    This class performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    The masked sequence is split into multiple chromosomes, each filled independently using crossover and mutation operations.
    During crossover, each chromosome from "num_parents" parents are crossed over independently, and the resulting child chromosomes are merged to form the child sequences.
    The population is divided into multiple islands, each evolving independently with occasional gene flow between them.

    """

    def __init__(
            self,
            cnn_model_path,
            masked_sequence,
            target_expression,
            precision=0.001,
            max_length=150,
            pop_size=100,
            generations=100, 
            base_mutation_rate=0.05,
            chromosomes=1,
            islands=1,
            gene_flow_rate=0.5,
            surval_rate=0.5,
            num_parents=2,
            num_competitors=5,
            selection='tournament',
            boltzmann_temperature=1,
            print_progress=True,
            early_stopping=True
    ):
        self.device = self.get_device()
        self.cnn = load_model(cnn_model_path)
        self.masked_sequence = masked_sequence
        self.target_expression = target_expression
        self.precision = precision
        self.max_length = max_length
        self.pop_size = pop_size
        self.generations = generations
        self.base_mutation_rate = base_mutation_rate
        self.chromosomes = chromosomes
        self.islands = islands
        self.gene_flow_rate = gene_flow_rate
        self.surviving_pop = max(1, int((self.pop_size / self.islands) * surval_rate)) # Ensure surviving_pop is at least 1
        self.num_parents = min(num_parents, self.surviving_pop) # Ensure num_parents is not larger than surviving_pop
        self.num_competitors = min(num_competitors, self.surviving_pop) # Ensure num_competitors is not larger than surviving_pop
        self.selection_method = self._get_selection_method(selection)
        self.boltzmann_temperature = boltzmann_temperature
        self.print_progress = print_progress
        self.early_stopping = early_stopping
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.chromosome_lengths = self._split_chromosome_lengths(self.mask_length, chromosomes)
        self.island_populations = [self.initialize_infills(self.mask_length, pop_size // islands) for _ in range(islands)]
        self.best_island_sequences = [None] * islands
        self.best_island_fitnesses = [-float('inf')] * islands
        self.best_island_predictions = [None] * islands

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_selection_method(self, selection):
        methods = {
            'tournament': self._select_tournament,
            'tournament_pop': self._select_tournament_pop,
            'roulette': self._select_roulette,
            'rank_based': self._select_rank_based,
            'truncation': self._select_truncation,
            'boltzmann': self._select_boltzmann,
        }
        if selection not in methods:
            raise ValueError(f"Invalid selection method: {selection}")
        return methods[selection]
    
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
            
    def _select_tournament(self, population, fitness_scores):
        parents = []
        for _ in range(self.surviving_pop):
            competitors = random.sample(range(len(population)), k=self.num_competitors)
            winner = max(competitors, key=lambda idx: fitness_scores[idx])
            parents.append(population[winner])
        return parents
    
    def _select_tournament_pop(self, population, fitness_scores):
        remaining_population = list(population)
        remaining_fitness_scores = list(fitness_scores)
        parents = []
        for _ in range(self.surviving_pop):
            competitors = random.sample(range(len(remaining_population)), k=self.num_competitors)
            winner_idx = max(competitors, key=lambda idx: remaining_fitness_scores[idx])
            parents.append(remaining_population[winner_idx])
            del remaining_population[winner_idx]
            del remaining_fitness_scores[winner_idx]
        return parents
    
    def _select_roulette(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = []
        for _ in range(self.surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def _select_rank_based(self, population, fitness_scores):
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx])
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        total_rank = sum(ranks.values())
        probabilities = [ranks[idx] / total_rank for idx in range(len(population))]
        parents = []
        for _ in range(self.surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def _select_truncation(self, population, fitness_scores):
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx], reverse=True)
        parents = [population[idx] for idx in sorted_indices[:self.surviving_pop]]
        return parents
    
    def _select_boltzmann(self, population, fitness_scores):
        boltzmann_scores = [math.exp(score / self.boltzmann_temperature) for score in fitness_scores]
        total_score = sum(boltzmann_scores)
        probabilities = [score / total_score for score in boltzmann_scores]
        parents = []
        for _ in range(self.surviving_pop):
            pick = random.uniform(0, 1)
            cumulative = 0
            for idx, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    parents.append(population[idx])
                    break
        return parents

    def recombination(self, parents):
        parent_chromosomes = [self._split_into_chromosomes(parent) for parent in parents]
        child_chromosomes = []
        for chrom_idx in range(len(parent_chromosomes[0])):
            chrom_slices = [parent[chrom_idx] for parent in parent_chromosomes]
            child_chromosome = self.crossover(chrom_slices)
            child_chromosomes.append(child_chromosome)

        return self._merge_chromosomes(child_chromosomes)
    
    @staticmethod
    def crossover(chrom_slices):
        if len(chrom_slices) == 1:
            return chrom_slices[0]
        if len(chrom_slices[0]) == 1:
            return random.choice(chrom_slices)
        if len(chrom_slices[0]) == 2:
            parent1, parent2 = random.sample(chrom_slices, 2)
            return parent1[:1] + parent2[1:]
        
        crossover_point = random.randint(1, len(chrom_slices[0]) - 1)
        parent1, parent2 = random.sample(chrom_slices, 2)
        return parent1[:crossover_point] + parent2[crossover_point:]

    @staticmethod
    def mutate(infill, mutation_rate=0.1):
        infill = list(infill)
        for i in range(len(infill)):
            if random.random() < mutation_rate:
                infill[i] = random.choice(['A', 'C', 'G', 'T'])
        return ''.join(infill)

    def gene_flow(self):
        if self.islands == 1 or self.gene_flow_rate == 0:
            return
        
        for i in range(self.islands):
            # Select a random island to exchange individuals with
            donor_island = random.choice([j for j in range(self.islands) if j != i])
            num_individuals = int(self.gene_flow_rate * len(self.island_populations[i]))

            # Select individuals to migrate
            migrants_to_island = random.sample(self.island_populations[donor_island], num_individuals)
            migrants_from_island = random.sample(self.island_populations[i], num_individuals)

            # Exchange individuals
            self.island_populations[i].extend(migrants_to_island)
            self.island_populations[donor_island].extend(migrants_from_island)

            # Ensure populations do not exceed original size
            self.island_populations[i] = random.sample(self.island_populations[i], len(self.island_populations[i]) - num_individuals)
            self.island_populations[donor_island] = random.sample(self.island_populations[donor_island], len(self.island_populations[donor_island]) - num_individuals)

    def run(self):
        for gen in range(self.generations):
            for i, infills in enumerate(self.island_populations):
                fitness_scores, predictions = self.evaluate_population(infills)

                best_idx = np.argmax(fitness_scores)
                if fitness_scores[best_idx] > self.best_island_fitnesses[i]:
                    self.best_island_fitnesses[i] = fitness_scores[best_idx]
                    self.best_island_sequences[i] = infills[best_idx]
                    self.best_island_predictions[i] = predictions[best_idx]

                if self.print_progress:
                    best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_island_sequences[i], self.mask_indices)
                    print(f"Island {i+1}, Generation {gen+1} | Best TX rate: {self.best_island_predictions[i]:.4f} | Target TX rate: {self.target_expression} | Sequence: {best_sequence}")

                if self.early_stopping and abs(self.best_island_predictions[i] - self.target_expression) < self.precision:
                    if self.print_progress:
                        print(f"Island {i+1}: Early stopping as target TX rate is achieved.")
                    continue

                parents = self.selection_method(infills, fitness_scores)
                next_gen = []
                while len(next_gen) < len(infills):
                    selected_parents = random.sample(parents, self.num_parents)
                    child = self.recombination(selected_parents)
                    next_gen.append(self.mutate(child, self.base_mutation_rate))
                self.island_populations[i] = next_gen[:len(infills)]

            # Perform gene flow between islands
            self.gene_flow()

        overall_best_idx = np.argmax(self.best_island_fitnesses)
        best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_island_sequences[overall_best_idx], self.mask_indices)
        return best_sequence, self.best_island_predictions[overall_best_idx]


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
        num_parents=1,
        islands=1,
        gene_flow_rate=0,
        print_progress=True
    )
    best_sequence, best_prediction = ga.run()
    print("\nBest infilled sequence:", best_sequence)
    print("Predicted transcription rate:", best_prediction)