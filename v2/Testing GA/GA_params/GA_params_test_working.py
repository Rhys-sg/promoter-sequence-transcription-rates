import torch
import numpy as np
import random
import re
import math
from keras.models import load_model  # type: ignore


class GeneticAlgorithm:
    '''
    This class performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    The masked sequence is split into multiple chromosomes, each filled independently using crossover and mutation operations.
    During crossover, each chromosome from 'num_parents' parents are crossed over independently, and the resulting child chromosomes are merged to form the child sequences.
    The population is divided into multiple islands, each evolving independently with occasional gene flow between them.

    '''

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
            elitist_rate=0,
            repeat_avoidance_rate=1,
            islands=1,
            gene_flow_rate=0,
            surval_rate=0.5,
            num_parents=2,
            num_competitors=5,
            selection='tournament',
            boltzmann_temperature=1,
            print_progress=True,
            early_stopping=True,
            caching=True,
            seed=None
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
        self.elitist_rate = elitist_rate
        self.repeat_avoidance_rate = repeat_avoidance_rate
        self.islands = islands
        self.gene_flow_rate = gene_flow_rate
        self.surviving_pop = max(1, int((self.pop_size / self.islands) * surval_rate)) # Ensure surviving_pop is at least 1
        self.num_parents = min(num_parents, self.surviving_pop) # Ensure num_parents is not larger than surviving_pop
        self.selection_method = getattr(SelectionMethod(self.surviving_pop, elitist_rate, num_competitors, boltzmann_temperature), selection)
        self.print_progress = print_progress
        self.early_stopping = early_stopping
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.chromosome_lengths = self.split_chromosome_lengths(self.mask_length, chromosomes)

        # For tracking and memoization purposes, could use lru_cache instead
        self.caching = caching
        self.seen_sequences = {}

        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    @staticmethod
    def get_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def split_chromosome_lengths(self, total_length, chromosomes):
        '''Split the mask length into chromosome lengths.'''
        base_length = total_length // chromosomes
        lengths = [base_length] * chromosomes
        for i in range(total_length % chromosomes):
            lengths[i] += 1
        return lengths
    
    def run(self, lineages=1):
        best_sequences = []
        best_predictions = []
        lineage_island_pop_history = []
        for lineage_idx in range(lineages):
            lineage = Lineage(
                self,
                lineage_idx = lineage_idx,
            )

            # Run the genetic algorithm for the current lineage, record results
            best_sequence, best_prediction, island_pop_history = lineage.run()
            best_sequences.append(best_sequence)
            best_predictions.append(best_prediction)
            lineage_island_pop_history.append(island_pop_history)

        return best_sequences, best_predictions, island_pop_history

class Lineage:
    '''
    This class represents a lineage of individuals that evolve independently in one instance of the genetic algorithm.
    It is used to encapsulate running the algorithm multiple times to evaluate multiple generated sequences.
    It also uses hamming distance to ensure that subsequent lineage explore untapped sections of the 'sequence landscapes'
    
    '''
    def __init__(
            self,
            geneticAlgorithm,
            lineage_idx,
    ):
        self.geneticAlgorithm = geneticAlgorithm
        self.lineage_idx = lineage_idx
        self.device = self.geneticAlgorithm.device

        # Initialize population history for each island, starting with the initial population   
        self.island_pop_history = [[self.initialize_infills(self.geneticAlgorithm.mask_length, self.geneticAlgorithm.pop_size // self.geneticAlgorithm.islands)
                                    for _ in range(self.geneticAlgorithm.islands)]]
        self.current_island_pop = self.island_pop_history[0]

        # For tracking the best sequence and prediction for each island
        self.best_island_sequences = [None] * self.geneticAlgorithm.islands
        self.best_island_fitnesses = [-float('inf')] * self.geneticAlgorithm.islands
        self.best_island_predictions = [None] * self.geneticAlgorithm.islands
    
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
            self.reconstruct_sequence(self.geneticAlgorithm.masked_sequence, infill, self.geneticAlgorithm.mask_indices)
            for infill in infills
        ]
        if self.geneticAlgorithm.caching:
            to_evaluate = [seq for seq in full_population if seq not in self.geneticAlgorithm.seen_sequences]
        else:
            to_evaluate = full_population
        if to_evaluate:
            one_hot_pop = [self.one_hot_sequence(seq.zfill(self.geneticAlgorithm.max_length)) for seq in to_evaluate]
            one_hot_tensor = torch.tensor(np.stack(one_hot_pop), dtype=torch.float32)
            with torch.no_grad():
                predictions = self.geneticAlgorithm.cnn(one_hot_tensor).cpu().numpy().flatten()
            fitness_scores = -np.abs(predictions - self.geneticAlgorithm.target_expression)
            for seq, fitness, pred in zip(to_evaluate, fitness_scores, predictions):
                self.geneticAlgorithm.seen_sequences[seq] = (fitness, pred)
        fitness_scores = np.array([self.geneticAlgorithm.seen_sequences[seq][0] for seq in full_population])
        predictions = np.array([self.geneticAlgorithm.seen_sequences[seq][1] for seq in full_population])
        return fitness_scores, predictions

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

    def run(self):
        for generation_idx in range(self.geneticAlgorithm.generations):
            for island_idx, infills in enumerate(self.current_island_pop):
                self.evaluate_and_update_best(generation_idx, island_idx, infills)
                
                if self.geneticAlgorithm.early_stopping and self.check_early_stopping(island_idx):
                    return self.finalize_run()

                self.current_island_pop[island_idx] = self.generate_next_generation(infills)

            if self.geneticAlgorithm.islands > 1 and self.geneticAlgorithm.gene_flow_rate > 0:
                self.gene_flow()
            
            self.island_pop_history.append(list(self.current_island_pop))

        return self.finalize_run()

    def evaluate_and_update_best(self, generation_idx, island_idx, infills):
        fitness_scores, predictions = self.evaluate_population(infills)
        best_idx = np.argmax(fitness_scores)
        
        if fitness_scores[best_idx] > self.best_island_fitnesses[island_idx]:
            self.best_island_fitnesses[island_idx] = fitness_scores[best_idx]
            self.best_island_sequences[island_idx] = infills[best_idx]
            self.best_island_predictions[island_idx] = predictions[best_idx]

        if self.geneticAlgorithm.print_progress:
            best_sequence = self.reconstruct_sequence(
                self.geneticAlgorithm.masked_sequence,
                self.best_island_sequences[island_idx],
                self.geneticAlgorithm.mask_indices
            )
            print(
                f"Lineage {self.lineage_idx+1}, Island {island_idx+1}, "
                f"Generation {generation_idx+1} | Best TX rate: {self.best_island_predictions[island_idx]:.4f} | "
                f"Target TX rate: {self.geneticAlgorithm.target_expression} | Sequence: {best_sequence}"
            )

    def check_early_stopping(self, island_idx):
        if abs(self.best_island_predictions[island_idx] - self.geneticAlgorithm.target_expression) < self.geneticAlgorithm.precision:
            if self.geneticAlgorithm.print_progress:
                print(f"Island {island_idx+1}: Early stopping as target TX rate is achieved.")
            return True
        return False

    def generate_next_generation(self, infills):
        fitness_scores, _ = self.evaluate_population(infills)
        parents = self.geneticAlgorithm.selection_method(infills, fitness_scores, self.geneticAlgorithm.surviving_pop)
        next_gen = []

        while len(next_gen) < len(infills):
            selected_parents = random.sample(parents, self.geneticAlgorithm.num_parents)
            child = self.mutate(self.recombination(selected_parents), self.geneticAlgorithm.base_mutation_rate)

            if child not in self.geneticAlgorithm.seen_sequences or random.random() > self.geneticAlgorithm.repeat_avoidance_rate:
                next_gen.append(child)
        
        return next_gen[:len(infills)]
    
    def gene_flow(self):        
        for i in range(self.geneticAlgorithm.islands):
            # Select a random island to exchange individuals with
            donor_island = random.choice([j for j in range(self.geneticAlgorithm.islands) if j != i])
            num_individuals = int(self.geneticAlgorithm.gene_flow_rate * len(self.current_island_pop[i]))

            # Select individuals to migrate
            migrants_to_island = random.sample(self.current_island_pop[donor_island], num_individuals)
            migrants_from_island = random.sample(self.current_island_pop[i], num_individuals)

            # Exchange individuals
            self.current_island_pop[i].extend(migrants_to_island)
            self.current_island_pop[donor_island].extend(migrants_from_island)

            # Ensure pop do not exceed original size
            self.current_island_pop[i] = random.sample(self.current_island_pop[i], len(self.current_island_pop[i]) - num_individuals)
            self.current_island_pop[donor_island] = random.sample(self.current_island_pop[donor_island], len(self.current_island_pop[donor_island]) - num_individuals)

    def finalize_run(self):
        overall_best_idx = np.argmax(self.best_island_fitnesses)
        best_sequence = self.reconstruct_sequence(
            self.geneticAlgorithm.masked_sequence,
            self.best_island_sequences[overall_best_idx],
            self.geneticAlgorithm.mask_indices
        )
        return best_sequence, self.best_island_predictions[overall_best_idx], self.island_pop_history

    
class SelectionMethod():
    '''
    This class implements various selection methods for genetic algorithms and stores selection parameters.
    '''
    def __init__(self, surviving_pop, elitist_rate, num_competitors, boltzmann_temperature):
        self.elitist_rate = elitist_rate
        self.num_competitors = min(num_competitors, surviving_pop) # Ensure num_competitors is not larger than surviving_pop
        self.boltzmann_temperature = boltzmann_temperature
    
    def tournament(self, population, fitness_scores, surviving_pop):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected.'''
        parents = self.truncation(population, fitness_scores, max(1, int(self.elitist_rate * surviving_pop))) if self.elitist_rate > 0 else []
        for _ in range(surviving_pop):
            competitors = random.sample(range(len(population)), k=self.num_competitors)
            winner = max(competitors, key=lambda idx: fitness_scores[idx])
            parents.append(population[winner])
        return parents
    
    def tournament_pop(self, population, fitness_scores, surviving_pop):
        '''A group of individuals is randomly chosen from the population, and the one with the highest fitness is selected and removed from future tournaments.'''
        remaining_population = list(population)
        remaining_fitness_scores = list(fitness_scores)
        parents = []
        for _ in range(surviving_pop):
            competitors = random.sample(range(len(remaining_population)), k=self.num_competitors)
            winner_idx = max(competitors, key=lambda idx: remaining_fitness_scores[idx])
            parents.append(remaining_population[winner_idx])
            del remaining_population[winner_idx]
            del remaining_fitness_scores[winner_idx]
        return parents
    
    def roulette(self, population, fitness_scores, surviving_pop):
        ''''Individuals are selected with a probability proportional to their fitness.'''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
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
    
    def linear_scaling(self, population, fitness_scores, surviving_pop):
        '''Fitness scores are normalized, and then roulette selection is performed.'''
        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)
        adjusted_scores = [(score - min_fitness) / (max_fitness - min_fitness + 1e-6) for score in fitness_scores]
        return self.roulette(population, adjusted_scores)
    
    def rank_based(self, population, fitness_scores, surviving_pop):
        '''Individuals are ranked based on their fitness, and selection probabilities are assigned based on rank rather than absolute fitness.'''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx])
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        total_rank = sum(ranks.values())
        probabilities = [ranks[idx] / total_rank for idx in range(len(population))]
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
    
    def sus(self, population, fitness_scores, surviving_pop):
        '''
        Similar to roulette wheel selection, but instead of selecting one individual at a time,
        Stochastic Universal Sampling (SUS) uses multiple equally spaced pointers to select individuals simultaneously.

        '''
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
        step = 1.0 / surviving_pop
        start = random.uniform(0, step)
        pointers = [start + i * step for i in range(surviving_pop)]
        
        parents = []
        for pointer in pointers:
            for idx, cumulative in enumerate(cumulative_probabilities):
                if pointer <= cumulative:
                    parents.append(population[idx])
                    break
        return parents
    
    def truncation(self, population, fitness_scores, surviving_pop):
        '''
        Only the top individuals are selected for the next generation.
        This method is reused for elitist selection by setting elitist_rate to a value between 0 and 1.
        '''
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda idx: fitness_scores[idx], reverse=True)
        parents = [population[idx] for idx in sorted_indices[:surviving_pop]]
        return parents
    
    def boltzmann(self, population, fitness_scores, surviving_pop):
        '''
        Based on simulated annealing, this method adjusts selection probabilities dynamically over time,
        favoring exploration in early generations and exploitation in later generations.
        
        '''
        boltzmann_scores = [math.exp(score / self.boltzmann_temperature) for score in fitness_scores]
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


if __name__ == '__main__':
    cnn_model_path = 'v2/Models/CNN_6_1_2.keras'
    masked_sequence = 'AATACTAGAGGTCTTCCGACNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNGTGTGGGCGGGAAGACAACTAGGGG'
    target_expression = 1

    ga = GeneticAlgorithm(
        cnn_model_path=cnn_model_path,
        masked_sequence=masked_sequence,
        target_expression=target_expression,
    )
    best_sequence, best_prediction, island_pop_history = ga.run(10)
    print('\nBest infilled sequence:', best_sequence)
    print('Predicted transcription rate:', best_prediction)