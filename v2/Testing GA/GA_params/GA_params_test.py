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
        self.previous_lineage_infills = {}
        self.seen_infills = {}

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

        for lineage_idx in range(lineages):
            lineage = Lineage(self, lineage_idx)

            # Run the genetic algorithm for the current lineage
            best_sequence, best_prediction = lineage.run()

            # Update the seen infills with the best infill from the current lineage
            self.previous_lineage_infills.update(self.seen_infills)

            best_sequences.append(best_sequence)
            best_predictions.append(best_prediction)

        return best_sequences, best_predictions

class Lineage:
    def __init__(self, geneticAlgorithm, lineage_idx):
        self.geneticAlgorithm = geneticAlgorithm
        self.lineage_idx = lineage_idx
        self.generation_idx = 0
        self.islands = [Island(self, geneticAlgorithm, island_idx) for island_idx in range(geneticAlgorithm.islands)]

        self.best_sequence = None
        self.best_fitness = -float('inf')
        self.best_prediction = None

    def run(self):
        while self.generation_idx < self.geneticAlgorithm.generations:
            for island in self.islands:
                island.population = island.generate_next_population()

            if self.geneticAlgorithm.islands > 1 and self.geneticAlgorithm.gene_flow_rate > 0:
                self.apply_gene_flow()

            self.update_best()

            if self.geneticAlgorithm.early_stopping and self.check_early_stopping():
                break

            self.generation_idx += 1

        return self.finalize_run()
    
    def apply_gene_flow(self):        
        for recipient_idx in range(self.islands):
            # Select a random island to exchange individuals with
            donor_idx = random.choice([j for j in range(self.islands) if j != recipient_idx])
            num_individuals = int(self.geneticAlgorithm.gene_flow_rate * len(self.islands[recipient_idx].population))

            # Select individuals to migrate
            migrants_to_island = random.sample(self.islands[donor_idx].population, num_individuals)
            migrants_from_island = random.sample(self.islands[recipient_idx].population, num_individuals)

            # Exchange individuals
            self.islands[recipient_idx].population.extend(migrants_to_island)
            self.islands[donor_idx].population.extend(migrants_from_island)

            # Ensure pop do not exceed original size
            self.islands[recipient_idx].population = random.sample(self.islands[recipient_idx].population, len(self.islands[recipient_idx].population) - num_individuals)
            self.islands[donor_idx].population = random.sample(self.islands[donor_idx].population, len(self.islands[donor_idx].population) - num_individuals)

    def update_best(self):
        for island in self.islands:
            if island.best_fitness > self.best_fitness:
                self.best_fitness = island.best_fitness
                self.best_sequence = island.best_sequence
                self.best_prediction = island.best_prediction

    def check_early_stopping(self):
        if abs(self.best_prediction - self.geneticAlgorithm.target_expression) < self.geneticAlgorithm.precision:
            if self.geneticAlgorithm.print_progress:
                print(f'Lineage {self.idx+1}: Early stopping as target TX rate is achieved.')
            return True
        return False

    def finalize_run(self):
        return self.best_sequence, self.best_prediction       

class Island:
    def __init__(self, lineage, geneticAlgorithm, idx):
        self.lineage = lineage
        self.geneticAlgorithm = geneticAlgorithm
        self.idx = idx
        self.population = self.initialize_infills()

        self.best_sequence = None
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
        skipped_children = 0

        while len(next_gen) < len(self.population):
            selected_parents = random.sample(parents, self.geneticAlgorithm.num_parents)
            child = self.mutate(self.recombination(selected_parents), self.geneticAlgorithm.base_mutation_rate)
            if child not in self.geneticAlgorithm.previous_lineage_infills or random.random() > self.geneticAlgorithm.repeat_avoidance_rate:
                next_gen.append(child)
            else:
                skipped_children += 1

        self.update_best(fitness_scores, predictions)
        if self.geneticAlgorithm.print_progress:
            self.print_progress(fitness_scores, predictions, skipped_children)
        
        return next_gen[:len(self.population)]
    
    def evaluate_population(self, infills):
        if self.geneticAlgorithm.caching:
            to_evaluate_infills = [infill for infill in infills if infill not in self.geneticAlgorithm.seen_infills]
        else:
            to_evaluate_infills = infills

        if to_evaluate_infills:
            to_evaluate = [
                self.reconstruct_sequence(self.geneticAlgorithm.masked_sequence, infill, self.geneticAlgorithm.mask_indices)
                for infill in to_evaluate_infills
            ]
            one_hot_pop = [self.one_hot_sequence(seq.zfill(self.geneticAlgorithm.max_length)) for seq in to_evaluate]
            one_hot_tensor = torch.tensor(np.stack(one_hot_pop), dtype=torch.float32)
            with torch.no_grad():
                predictions = self.geneticAlgorithm.cnn(one_hot_tensor).cpu().numpy().flatten()
            fitness_scores = -np.abs(predictions - self.geneticAlgorithm.target_expression)
            for infill, fitness, pred in zip(to_evaluate_infills, fitness_scores, predictions):
                self.geneticAlgorithm.seen_infills[infill] = (fitness, pred)
        fitness_scores = np.array([self.geneticAlgorithm.seen_infills[infill][0] for infill in infills])
        predictions = np.array([self.geneticAlgorithm.seen_infills[infill][1] for infill in infills])
        return fitness_scores, predictions
    
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
    
    def update_best(self, fitness_scores, predictions):
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_sequence = self.population[best_idx]
            self.best_prediction = predictions[best_idx]
            
    def print_progress(self, fitness_scores, predictions, skipped_children):
        if self.geneticAlgorithm.print_progress:
            best_sequence = self.reconstruct_sequence(
                self.geneticAlgorithm.masked_sequence,
                self.best_sequence,
                self.geneticAlgorithm.mask_indices
            )
            print(
                f'Lineage {self.lineage.lineage_idx+1} | ' +
                f'Island {self.idx+1} | ' +
                f'Generation {self.lineage.generation_idx+1} | ' +
                f'Best TX rate: {self.best_prediction:.4f} | ' +
                f'Sequence: {best_sequence} | ' +
                f'Children Not Added: {skipped_children}'
            )

    @staticmethod
    def reconstruct_sequence(base_sequence, infill, mask_indices):
        sequence = list(base_sequence)
        for idx, char in zip(mask_indices, infill):
            sequence[idx] = char
        return ''.join(sequence)
    
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
    best_sequences, best_predictions = ga.run(3)
    print('\nBest infilled sequences:', best_sequences)
    print('Predicted transcription rates:', best_predictions)