import torch
import numpy as np
import random
import re
from keras.models import load_model  # type: ignore

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time


class GeneticAlgorithm:
    """
    This class performs genetic algorithm to infill a masked sequence with nucleotides that maximize the predicted transcription rate.
    The fitness of each individual is calculated as the negative absolute difference between the predicted transcription rate and the target rate.
    The surviving population is selected using tournament selection, and the next generation is created using crossover and mutation.
    It considers just the infilled sequence, not the entire sequence. The infill is mutated, crossed over, and then the entire sequence is
    reconstructed before the sequence is selected based on fitness.

    Includes chromosomes, multiple parents, multiple competitors for tournament selection, and seperate lineages.

    """

    def __init__(self, cnn_model_path, masked_sequence, target_expression, max_length=150, pop_size=20, generations=100,
             base_mutation_rate=0.05, precision=0.01, chromosomes=1, num_parents=2, num_competitors=5, print_progress=True):
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
        self.num_parents = num_parents
        self.num_competitors = min(num_competitors, pop_size)
        self.print_progress = print_progress
        self.mask_indices = [i for i, nucleotide in enumerate(masked_sequence) if nucleotide == 'N']
        self.mask_length = len(self.mask_indices)
        self.chromosome_lengths = self._split_chromosome_lengths(self.mask_length, chromosomes)
        self.best_infill = None

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
    def select_parents(population, fitness_scores, num_parents, num_competitors):
        parents = []
        for _ in range(num_parents):
            competitors = random.sample(range(len(population)), k=num_competitors)
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

            parents = self.select_parents(infills, fitness_scores, self.num_parents, self.num_competitors)
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

class GeneticAlgorithmWithVisualization(GeneticAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []  # To store fitness history for visualization

    def run_lineage(self, lineage):
        infills = self.initialize_infills(self.mask_length, self.pop_size)
        lineage_history = []  # Store history for this lineage

        self.best_fitness = -float('inf')
        self.best_infill = None
        self.best_prediction = 0

        for gen in range(self.generations):
            fitness_scores, predictions = self.evaluate_population(infills)

            # Record fitness scores for visualization
            lineage_history.append((gen, infills, fitness_scores))

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

            parents = self.select_parents(infills, fitness_scores, self.num_parents, self.num_competitors)
            next_gen = []
            while len(next_gen) < self.pop_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                next_gen.append(self.mutate(child1, self.base_mutation_rate))
                next_gen.append(self.mutate(child2, self.base_mutation_rate))
            infills = next_gen

        # Add the lineage history to global history
        self.history.append(lineage_history)

        best_sequence = self.reconstruct_sequence(self.masked_sequence, self.best_infill, self.mask_indices)
        return best_sequence, self.best_prediction

    def visualize(self, top_n=10):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for lineage_idx, lineage_history in enumerate(self.history):
            generations = []
            fitness_values = []
            x_similarity = []
            y_similarity = []

            for gen, infills, fitness_scores in lineage_history:
                # Get indices of top n sequences by fitness
                top_indices = np.argsort(fitness_scores)[-top_n:]

                # Choose a reference infill for similarity calculations (e.g., the best infill)
                reference_infill = infills[np.argmax(fitness_scores)]

                # Split the reference infill into two halves
                half_length = len(reference_infill) // 2
                ref_first_half = reference_infill[:half_length]
                ref_second_half = reference_infill[half_length:]

                # Compute Hamming similarity for each half
                for i in top_indices:
                    infill = infills[i]
                    infill_first_half = infill[:half_length]
                    infill_second_half = infill[half_length:]

                    # Hamming similarity for the first half
                    x_sim = 1 - sum(a != b for a, b in zip(infill_first_half, ref_first_half)) / half_length

                    # Hamming similarity for the second half
                    y_sim = 1 - sum(a != b for a, b in zip(infill_second_half, ref_second_half)) / half_length

                    x_similarity.append(x_sim)
                    y_similarity.append(y_sim)
                    fitness_values.append(fitness_scores[i])

                generations.extend([gen] * len(top_indices))

            # Plot the top n sequences for this lineage
            ax.scatter(
                x_similarity,
                y_similarity,
                fitness_values,
                label=f'Lineage {lineage_idx+1}',
                alpha=0.7
            )

        ax.set_xlabel("Hamming Similarity (First Half, X)")
        ax.set_ylabel("Hamming Similarity (Second Half, Y)")
        ax.set_zlabel("Fitness")
        ax.legend()
        plt.show()

    def dynamic_visualize(self, top_n=10, delay=0.1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Data placeholders
        all_generations = []
        all_x_similarity = []
        all_y_similarity = []
        all_fitness_values = []

        # Collect data and determine axis limits
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')

        max_generations = 0

        for lineage_history in self.history:
            generations = []
            x_similarity = []
            y_similarity = []
            fitness_values = []

            for gen, infills, fitness_scores in lineage_history:
                # Get indices of top n sequences by fitness
                top_indices = np.argsort(fitness_scores)[-top_n:]

                # Choose a reference infill for similarity calculations (e.g., the best infill)
                reference_infill = infills[np.argmax(fitness_scores)]

                # Split the reference infill into two halves
                half_length = len(reference_infill) // 2
                ref_first_half = reference_infill[:half_length]
                ref_second_half = reference_infill[half_length:]

                # Compute Hamming similarity for each half
                for i in top_indices:
                    infill = infills[i]
                    infill_first_half = infill[:half_length]
                    infill_second_half = infill[half_length:]

                    # Hamming similarity for the first half
                    x_sim = 1 - sum(a != b for a, b in zip(infill_first_half, ref_first_half)) / half_length

                    # Hamming similarity for the second half
                    y_sim = 1 - sum(a != b for a, b in zip(infill_second_half, ref_second_half)) / half_length

                    x_similarity.append(x_sim)
                    y_similarity.append(y_sim)
                    fitness_values.append(fitness_scores[i])

                    # Update axis limits
                    x_min, x_max = min(x_min, x_sim), max(x_max, x_sim)
                    y_min, y_max = min(y_min, y_sim), max(y_max, y_sim)
                    z_min, z_max = min(z_min, fitness_scores[i]), max(z_max, fitness_scores[i])

                generations.extend([gen] * len(top_indices))

            max_generations = max(max_generations, max(generations) + 1)
            all_generations.append(generations)
            all_x_similarity.append(x_similarity)
            all_y_similarity.append(y_similarity)
            all_fitness_values.append(fitness_values)

        # Define the animation update function
        def update(frame):
            ax.clear()
            ax.set_xlabel("Hamming Similarity (First Half, X)")
            ax.set_ylabel("Hamming Similarity (Second Half, Y)")
            ax.set_zlabel("Fitness")
            ax.set_title(f"Generation {frame + 1}")

            # Set fixed axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)

            has_data = False
            for lineage_idx, (generations, x_similarity, y_similarity, fitness_values) in enumerate(
                zip(all_generations, all_x_similarity, all_y_similarity, all_fitness_values)):
                indices = [i for i, gen in enumerate(generations) if gen == frame]
                if indices:
                    ax.scatter(
                        [x_similarity[i] for i in indices],
                        [y_similarity[i] for i in indices],
                        [fitness_values[i] for i in indices],
                        label=f'Lineage {lineage_idx+1}',
                        alpha=0.7
                    )
                    has_data = True

            if has_data:
                ax.legend()

        # Create animation
        ani = FuncAnimation(fig, update, frames=max_generations, interval=delay * 1000, repeat=True)

        plt.show()



if __name__ == '__main__':
    cnn_model_path = 'v2/Models/CNN_6_1_2.keras'
    masked_sequence = 'NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN'
    target_expression = 0.9

    ga = GeneticAlgorithmWithVisualization(
        cnn_model_path=cnn_model_path,
        masked_sequence=masked_sequence,
        target_expression=target_expression,
        pop_size=100,
        generations=100,
        base_mutation_rate=0.05,
        precision=0.001,
        chromosomes=3,
        print_progress=True
    )
    best_sequences, best_predictions = ga.run(3)

    print("\nBest infilled sequence:", best_sequences)
    print("Predicted transcription rate:", best_predictions)

    ga.dynamic_visualize(top_n=3)
