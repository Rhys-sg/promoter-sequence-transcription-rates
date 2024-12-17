import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from IPython.display import HTML

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def plot_3d_high_fitness_individuals_animated(fitness_histories, population_histories, n_top=5):
    lineage_data = []
    lineage_colors = plt.cm.tab10(np.linspace(0, 1, len(fitness_histories)))

    # Prepare lineage data
    for lineage_idx, (lineage_fitness_history, lineage_population_history) in enumerate(zip(fitness_histories, population_histories)):
        lineage = []
        for generation_fitness_history, generation_population_history in zip(lineage_fitness_history, lineage_population_history):
            generation_data = []
            for island_fitnesses, island_population in zip(generation_fitness_history, generation_population_history):
                for fitnesses, population in zip(island_fitnesses, island_population):
                    combined = list(zip(fitnesses, population))
                    combined.sort(reverse=True, key=lambda x: x[0])  # Sort by fitness descending
                    generation_data.extend([(lineage_idx, fitness, seq) for fitness, seq in combined[:n_top]])
            lineage.append(generation_data)
        lineage_data.append(lineage)

    # Collect all sequences for PCA
    all_sequences = {seq for lineage in lineage_data for generation in lineage for _, _, seq in generation}
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    kmer_features = vectorizer.fit_transform(all_sequences).toarray()

    # Perform PCA
    pca = PCA(n_components=2)
    pca_transformed = pca.fit_transform(kmer_features)
    seq_to_pca = {seq: coord for seq, coord in zip(all_sequences, pca_transformed)}

    # Initialize plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(generation_idx):
        ax.cla()  # Clear axes without resetting them
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.set_zlabel("Fitness")
        ax.set_title(f"3D Fitness Plot (Generation {generation_idx + 1})")

        for lineage_idx, lineage in enumerate(lineage_data):
            if generation_idx < len(lineage):
                generation = lineage[generation_idx]
                x = [seq_to_pca[seq][0] for _, _, seq in generation]
                y = [seq_to_pca[seq][1] for _, _, seq in generation]
                z = [fitness for _, fitness, _ in generation]
                ax.scatter(x, y, z, label=f"Lineage {lineage_idx + 1}", color=lineage_colors[lineage_idx])
        
        ax.legend()

    num_generations = max(len(lineage) for lineage in lineage_data)
    ani = FuncAnimation(fig, update, frames=num_generations, interval=1000)

    plt.show()
    

def hamming_distance(seq1, seq2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))

def plot_hamming_distance(lineage_island_pop_history):
    average_distances = []

    for lineage in lineage_island_pop_history:
        lineage_distances = []
        for lineage_population in lineage:
            if len(lineage_population) < 2:
                continue
            pairwise_distances = [
                hamming_distance(ind1, ind2)
                for i, ind1 in enumerate(lineage_population)
                for j, ind2 in enumerate(lineage_population)
                if i < j
            ]
            # Average Hamming distance for this lineage
            if pairwise_distances:
                lineage_distances.append(np.mean(pairwise_distances))
        # Average across all lineages
        if lineage_distances:
            average_distances.append(np.mean(lineage_distances))

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(average_distances)), average_distances, marker='o', linestyle='-', label='Average Hamming Distance')
    plt.title("Average Hamming Distance Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Hamming Distance")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fitness_histories(fitness_histories):
    for lineage_idx, lineage_history in enumerate(fitness_histories):
        for island_idx, island_history in enumerate(lineage_history):
            avg_fitness_per_gen = [np.mean(fitness_scores) for fitness_scores in island_history]
            plt.plot(avg_fitness_per_gen, label=f'Lineage {lineage_idx + 1}, Island {island_idx + 1}')

    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Fitness Over Generations')
    plt.legend()
    plt.show()

def plot_3d_fitness(fitness_histories, lineage_island_pop_history, top_n=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    reference_a = 'A' * len(lineage_island_pop_history[0][0][0][0])
    reference_c = 'C' * len(lineage_island_pop_history[0][0][0][0])
    print(len(reference_a), len(reference_c))
    
    for lineage_idx, (lineage_history, fitness_history) in enumerate(zip(lineage_island_pop_history, fitness_histories)):
        for island_idx, (island_history, island_fitness) in enumerate(zip(lineage_history, fitness_history)):
            for gen_idx, (gen_population, gen_fitness) in enumerate(zip(island_history, island_fitness)):
                # Select top N individuals by fitness
                sorted_indices = np.argsort(gen_fitness)[-top_n:]
                top_sequences = [gen_population[idx] for idx in sorted_indices]
                top_fitness = [gen_fitness[idx] for idx in sorted_indices]
                
                # Calculate Hamming distances
                x_values = [hamming_distance(seq, reference_a) for seq in top_sequences]
                z_values = [hamming_distance(seq, reference_c) for seq in top_sequences]
                
                # Plot points for this generation
                ax.scatter(x_values, top_fitness, z_values, label=f'Lineage {lineage_idx+1}, Island {island_idx+1}, Gen {gen_idx+1}')
    
    ax.set_xlabel('Hamming Distance to 75 A\'s')
    ax.set_ylabel('Fitness')
    ax.set_zlabel('Hamming Distance to 75 C\'s')
    plt.title('Top N Sequences by Fitness Across Generations')
    plt.legend()
    plt.show()
