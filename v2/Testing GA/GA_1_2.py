"""
This script performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
Each masked region is treated as a seperate chromosome, filled independently using crossover and mutation operations.
The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.

This approach includes adaptive mutation rate based on the diversity score of the population.
"""


import torch
import numpy as np
import random
import re
from keras.models import load_model

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def one_hot_sequence(seq):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0.25, 0.25, 0.25, 0.25], '0': [0, 0, 0, 0]}
    return np.array([mapping[nucleotide.upper()] for nucleotide in seq])

def find_masked_regions(sequence):
    """Identify all continuous masked regions in the sequence."""
    return [(m.start(), m.end()) for m in re.finditer('N+', sequence)]

def initialize_population(masked_sequence, mask_indices, pop_size):
    """Initialize a population by only filling 'N' positions with random nucleotides."""
    population = []
    for _ in range(pop_size):
        individual = list(masked_sequence)
        for i in mask_indices:
            individual[i] = random.choice(['A', 'C', 'G', 'T'])
        population.append(''.join(individual))
    return population

def evaluate_population(cnn, population, target_expression, max_length=150):
    """Evaluate the population by calculating the fitness of each individual."""
    one_hot_pop = [one_hot_sequence(seq.zfill(max_length)) for seq in population]
    one_hot_tensor = torch.tensor(np.stack(one_hot_pop), dtype=torch.float32)
    with torch.no_grad():
        predictions = cnn(one_hot_tensor).cpu().numpy().flatten()
    fitness_scores = -np.abs(predictions - target_expression)  # Maximize by minimizing the distance to target
    return fitness_scores, predictions

def select_parents(population, fitness_scores, num_parents):
    """Select parents based on fitness using tournament selection."""
    parents = []
    for _ in range(num_parents):
        competitors = random.sample(range(len(population)), k=5)  # Tournament size
        winner = max(competitors, key=lambda idx: fitness_scores[idx])
        parents.append(population[winner])
    return parents

def crossover(parent1, parent2, masked_regions):
    """Perform independent crossover in each masked region."""
    child1 = list(parent1)
    child2 = list(parent2)
    
    for region_start, region_end in masked_regions:
        # Masked region-specific indices
        region_indices = list(range(region_start, region_end))
        if len(region_indices) > 1:  # Ensure there's space for a crossover
            crossover_point = random.choice(region_indices)
            
            # Swap elements after the crossover point within the current masked region
            for i in region_indices:
                if i >= crossover_point:
                    child1[i], child2[i] = parent2[i], parent1[i]
                    
    return ''.join(child1), ''.join(child2)

def mutate(sequence, mask_indices, mutation_rate=0.1):
    """Mutate only the masked positions with a given mutation rate."""
    sequence = list(sequence)
    for i in mask_indices:
        if random.random() < mutation_rate:
            sequence[i] = random.choice(['A', 'C', 'G', 'T'])
    return ''.join(sequence)

def adapt_mutation_rate(mutation_rate, diversity_score, base_rate=0.1, max_rate=0.5, threshold=0.2):
    """Adapt the mutation rate based on diversity score.
       Increase mutation rate if diversity is below a threshold."""
    if diversity_score < threshold:
        return min(mutation_rate * 1.2, max_rate)
    return max(mutation_rate * 0.9, base_rate)

def calculate_diversity(population):
    """Calculate diversity score as the proportion of unique sequences in the population."""
    unique_sequences = len(set(population))
    diversity_score = unique_sequences / len(population)
    return diversity_score

def genetic_algorithm(cnn, masked_sequence, target_expression, pop_size=20, generations=100, base_mutation_rate=0.1, precision=0.01, print_progress=True):
    # Identify masked positions (positions with 'N')
    mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
    masked_regions = find_masked_regions(masked_sequence)

    # Initialize population
    population = initialize_population(masked_sequence, mask_indices, pop_size)
    best_sequence = None
    best_fitness = -float('inf')
    best_prediction = None

    for gen in range(generations):
        # Evaluate population
        fitness_scores, predictions = evaluate_population(cnn, population, target_expression)
        
        # Track the best sequence
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_sequence = population[best_idx]
            best_prediction = predictions[best_idx]
        
        if print_progress:
            print(f"Generation {gen+1} | Best TX rate: {best_prediction:.4f} | Target TX rate: {target_expression} | Sequence: {best_sequence}")
        
        # Early stopping if close to target
        if abs(best_prediction - target_expression) < precision:
            if print_progress:
                print("Early stopping as target TX rate is achieved.")
            break

        # Calculate diversity score for the island and adapt mutation rate
        diversity_score = calculate_diversity(population)
        mutation_rate = adapt_mutation_rate(base_mutation_rate, diversity_score)

        # Select parents and create next generation
        parents = select_parents(population, fitness_scores, pop_size // 2)
        next_gen = []
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2, masked_regions)
            next_gen.append(mutate(child1, mask_indices, mutation_rate))
            next_gen.append(mutate(child2, mask_indices, mutation_rate))
        
        population = next_gen

    return best_sequence, best_prediction

if __name__ == '__main__':

    # Paths and parameters
    path_to_cnn = 'v2/Models/CNN_6_1_2.keras'
    # masked_sequence = 'AATACTAGAGGTCTTCCGACNNNNNNATTAATCATCCGGCTCGNNNNNNNTGTGGAGCGGGAAGACAACTAGGGG'
    masked_sequence = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'
    target_expression = 1

    # Genetic algorithm parameters
    max_length = 150
    pop_size = 100
    generations = 100
    base_mutation_rate = 0.1
    precision = 0.001

    # Initialize CNN and device
    device = get_device()
    cnn = load_model(path_to_cnn)

    # Run GA to infill masked sequence
    best_sequence, best_prediction = genetic_algorithm(
        cnn, masked_sequence, target_expression, pop_size=pop_size, generations=generations, base_mutation_rate=base_mutation_rate, precision=precision
    )

    print("\nBest infilled sequence:", best_sequence)
    print("Predicted transcription rate:", best_prediction)
