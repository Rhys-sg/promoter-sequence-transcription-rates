"""
This script performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.

This approach considers just the infilled sequence as the chromosome, not the entire sequence. The infill is mutated, crossed over, and then
the entire sequence is reconstructed before the sequence is selected based on fitness.

"""

import torch
import numpy as np
import random
import re
from keras.models import load_model  # type: ignore


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def find_masked_regions(sequence):
    """Identify all continuous masked regions in the sequence."""
    return [(m.start(), m.end()) for m in re.finditer('N+', sequence)]


def initialize_infills(mask_length, pop_size):
    """Generate random infills of a given length."""
    return [
        ''.join([random.choice(['A', 'C', 'G', 'T']) for _ in range(mask_length)])
        for _ in range(pop_size)
    ]


def reconstruct_sequence(base_sequence, infill, mask_indices):
    """Reconstruct the full sequence by replacing masked positions with infill."""
    sequence = list(base_sequence)
    for idx, char in zip(mask_indices, infill):
        sequence[idx] = char
    return ''.join(sequence)


def evaluate_population(cnn, infills, target_expression, masked_sequence, mask_indices, max_length=150):
    """Evaluate the population of infills."""
    full_population = [
        reconstruct_sequence(masked_sequence, infill, mask_indices)
        for infill in infills
    ]
    one_hot_pop = [one_hot_sequence(seq.zfill(max_length)) for seq in full_population]
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


def crossover(parent1, parent2):
    """Perform crossover on two infills."""
    child1, child2 = list(parent1), list(parent2)
    if len(parent1) > 1:  # Ensure there's space for crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        child1[crossover_point:], child2[crossover_point:] = parent2[crossover_point:], parent1[crossover_point:]
    return ''.join(child1), ''.join(child2)


def mutate(infill, mutation_rate=0.1):
    """Mutate the infill with a given mutation rate."""
    infill = list(infill)
    for i in range(len(infill)):
        if random.random() < mutation_rate:
            infill[i] = random.choice(['A', 'C', 'G', 'T'])
    return ''.join(infill)


def genetic_algorithm(cnn, masked_sequence, target_expression, pop_size=20, generations=100, base_mutation_rate=0.1, precision=0.01, print_progress=True):
    # Identify masked positions (positions with 'N')
    mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
    mask_length = len(mask_indices)

    # Initialize population of infills
    infills = initialize_infills(mask_length, pop_size)

    best_infill = None
    best_fitness = -float('inf')
    best_prediction = None

    for gen in range(generations):
        # Evaluate population
        fitness_scores, predictions = evaluate_population(
            cnn, infills, target_expression, masked_sequence, mask_indices
        )

        # Track the best infill
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > best_fitness:
            best_fitness = fitness_scores[best_idx]
            best_infill = infills[best_idx]
            best_prediction = predictions[best_idx]

        if print_progress:
            best_sequence = reconstruct_sequence(masked_sequence, best_infill, mask_indices)
            print(f"Generation {gen+1} | Best TX rate: {best_prediction:.4f} | Target TX rate: {target_expression} | Sequence: {best_sequence}")

        # Early stopping if close to target
        if abs(best_prediction - target_expression) < precision:
            if print_progress:
                print("Early stopping as target TX rate is achieved.")
            break

        # Select parents and create next generation
        parents = select_parents(infills, fitness_scores, pop_size // 2)
        next_gen = []
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            next_gen.append(mutate(child1, base_mutation_rate))
            next_gen.append(mutate(child2, base_mutation_rate))
        infills = next_gen

    # Reconstruct the best sequence
    best_sequence = reconstruct_sequence(masked_sequence, best_infill, mask_indices)
    return best_sequence, best_prediction


if __name__ == '__main__':
    # Paths and parameters
    path_to_cnn = 'v2/Models/CNN_6_1_2.keras'
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
