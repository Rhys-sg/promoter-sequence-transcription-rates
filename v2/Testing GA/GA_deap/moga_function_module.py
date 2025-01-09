import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import itertools

from GA.MogaGeneticAlgorithm import MogaGeneticAlgorithm
from function_module import format_time

def test_params(param_ranges, target_expressions, lineages, kwargs, iteration=1):
    results = []
    initial_time = time.time()
    
    # Generate all combinations of parameters
    param_keys = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations) * len(target_expressions) * lineages
    current_combination = 0
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    
    for param_combination in param_combinations:
        params = dict(zip(param_keys, param_combination))
        
        for target_expression in target_expressions:
            for lineage in range(lineages):

                if 'seed' in kwargs:
                    kwargs['seed'] += 1
                
                kwargs = {**kwargs, **params}
                
                # Create genetic algorithm with the current parameter combination
                ga = MogaGeneticAlgorithm(**kwargs, target_expression=target_expression)

                start = time.time()
                ga.run()
                end = time.time()
                
                # Store results
                result = {
                    'target_expression': target_expression,
                    'lineage': lineage,
                    'sequence': ga.best_sequences[0],
                    'error': abs(target_expression - ga.best_predictions[0]),
                    'predictions': ga.best_predictions[0],
                    'run_time': end - start
                }
                results.append({**params, **result})

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

    results_df = pd.DataFrame(results)
    name = '_'.join(param_keys)
    results_df.to_csv(f'Data/{name}_results_{iteration}.csv', index=False)

    return results_df

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
    for lineage1, lineage2 in itertools.combinations(data, 2):
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