import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
from itertools import combinations
import json

from GA_params_class.GeneticAlgorithm import GeneticAlgorithm
from function_module import format_time

def test_divergence(lineage_divergence_alpha_range, cnn_model_path, masked_sequence, target_expressions, precision, verbose, lineage_range, iteration=1, seed=1):
    infill_histories = {
        lineages : {
            target_expression : {
                lineage_divergence_alpha : None
                for lineage_divergence_alpha in lineage_divergence_alpha_range}
            for target_expression in target_expressions}
        for lineages in lineage_range
    }
    results = []
    total_combinations = len(target_expressions) * len(lineage_divergence_alpha_range) * len(lineage_range)
    progress_bar = tqdm(total=total_combinations, desc="Processing combinations", position=0)
    initial_time = time.time()
    current_combination = 0

    for lineages in lineage_range:
        for target_expression in target_expressions:
            for i, lineage_divergence_alpha in enumerate(lineage_divergence_alpha_range):
                ga = GeneticAlgorithm(
                    cnn_model_path=cnn_model_path,
                    masked_sequence=masked_sequence,
                    target_expression=target_expression,
                    precision=precision,
                    verbose=verbose,
                    seed=seed,
                    lineage_divergence_alpha=lineage_divergence_alpha
                )
                # Time the run
                start_time = time.time()
                best_sequences, best_predictions = ga.run(lineages)
                end_time = time.time()

                infill_histories[lineages][target_expression][lineage_divergence_alpha] = ga.get_infill_history()

                min_index = best_predictions.index(min(best_predictions))
                results.append({
                    'lineages': lineages,
                    'target_expression': target_expression,
                    'lineage_divergence_alpha': lineage_divergence_alpha,
                    'sequence': best_sequences[min_index],
                    'error': abs(best_predictions[min_index] - target_expression),
                    'run_time': (end_time - start_time) / lineages
                })
                    
                # Update progress bar
                current_combination += 1
                progress_bar.update(1)
                elapsed_time = time.time() - initial_time
                progress_bar.set_postfix({
                    "Elapsed": format_time(elapsed_time),
                    "ETA": format_time(((elapsed_time / current_combination) * (total_combinations - current_combination)))
                })

    # Close progress bar
    progress_bar.close()

    # Save results to csv
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'Data/MOGA/{lineages}_lineages_results_{iteration}.csv', index=False)

    # # Save infill histories to json
    # with open(f'Data/lineage_divergence_alpha/{lineages}_lineages_infill_histories_{iteration}.json', 'w') as f:
    #     json.dump(infill_histories, f)
       
    return results_df, infill_histories

def divergence_scatter_plot(results_df, target_expression, index, polynomial_degree=1):
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # Create a color mapping for each lineage
    unique_lineages = results_df['lineages'].unique()
    colors = plt.cm.tab10(range(len(unique_lineages)))
    lineage_color_map = {lineage: (0, 0, 1 * (lineage/len(unique_lineages))) for lineage in unique_lineages}

    def add_best_fit_line(ax, x, y, degree, label, color):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color=color, linestyle='--')

    # Error vs index
    for lineage in unique_lineages:
        filter_df = results_df[results_df['lineages'] == lineage]
        axes[0].scatter(
            filter_df['lineage_divergence_alpha'],
            filter_df['error'],
            label=lineage,
            color=lineage_color_map[lineage],
            alpha=0.7
        )
        add_best_fit_line(axes[0], filter_df[index], filter_df['error'], polynomial_degree, '', lineage_color_map[lineage])
    axes[0].set_xlabel(index)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Minimum Error vs {index} with Target Expression {target_expression}')

    # Runtime vs index
    for lineage in unique_lineages:
        filter_df = results_df[results_df['lineages'] == lineage]
        axes[1].scatter(
            filter_df['lineage_divergence_alpha'],
            filter_df['run_time'],
            label=lineage,
            color=lineage_color_map[lineage],
            alpha=0.7
        )
        add_best_fit_line(axes[1], filter_df[index], filter_df['run_time'], polynomial_degree, f'', lineage_color_map[lineage])
    axes[1].set_xlabel(index)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index} with Target Expression {target_expression}')
    axes[1].legend(title='Lineages', bbox_to_anchor=(1.05, 1), loc='upper left')

def calculate_hamm_distance(str1, str2):
    return sum([1 for s, t in zip(str1, str2) if s != t])

def calculate_average_hamm_distance(pop1, pop2):
    # Calculate the average hamming distance between all pairs of sequences in two populations
    total_distance = 0
    pair_count = 0
    
    for i in range(len(pop1)):
        for j in range(len(pop2)):
            total_distance += calculate_hamm_distance(pop1[i], pop2[j])
            pair_count += 1
    
    return total_distance / pair_count if pair_count > 0 else 0

def calculate_intra_lineage_hamm(infill_history):
    # Group all individuals by generation, removing islands
    data = [[[] for generation in range(len(infill_history[0][0]))] for lineage in range(len(infill_history))]
    for lineage_idx, lineage in enumerate(infill_history):
        for island in lineage:
            for generation_idx, generation in enumerate(island):
                data[lineage_idx][generation_idx].extend(generation)

    # Calculate average hamming distance between each individual of DIFFERENT lineages, for each generation
    distances = [0 for generation in data[0]]
    for lineage1, lineage2 in combinations(data, 2):
        for generation_idx, generation in enumerate(lineage1):
            distances[generation_idx] += calculate_average_hamm_distance(generation, lineage2[generation_idx])
    return distances

def plot_intra_hamm_comparison(datas, lineage_divergence_alpha):
    plt.figure(figsize=(10, 6))
    for idx, data in enumerate(datas):
        generations = range(len(data))
        plt.plot(
            generations, 
            data, 
            label=f'Alpha = {lineage_divergence_alpha[idx]:.2f}', 
            color=(1, 0, 0, lineage_divergence_alpha[idx])  # RGBA for red with variable alpha
        )
    
    plt.xlabel('Generation')
    plt.ylabel('Average Hamming Distance')
    plt.legend()
    plt.show()