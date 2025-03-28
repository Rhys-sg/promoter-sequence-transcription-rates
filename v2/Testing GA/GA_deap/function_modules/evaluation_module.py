import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import itertools

from GA.CombinatorialAlgorithm import CombinatorialAlgorithm
from GA.GeneticAlgorithm import GeneticAlgorithm

from .function_module import format_time

def test_params(param_ranges, target_expressions, lineages, kwargs, to_csv=None, iteration=1):
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
                try:
                    if 'seed' in kwargs:
                        kwargs['seed'] += 1
                    
                    kwargs = {**kwargs, **params}
                    
                    # Create genetic algorithm with the current parameter combination
                    ga = GeneticAlgorithm(**kwargs, target_expression=target_expression)

                    start = time.time()
                    ga.run()
                    end = time.time()
                    
                    # Store results
                    result = {
                        'target_expression': target_expression,
                        'lineage': lineage,
                        'sequence': ga.best_sequence,
                        'error': abs(target_expression - ga.best_prediction),
                        'prediction': ga.best_prediction,
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
                except Exception as e:
                    print(f'Error: {e}')
                    print(f'Params: {params}, Target Expression: {target_expression}, Lineage: {lineage}')
                    continue
    
    # Close progress bar
    progress_bar.close()

    results_df = pd.DataFrame(results)
    if to_csv != None:
        results_df.to_csv(to_csv, index=False)

    return results_df

def test_combinatorial(masked_sequences, target_expressions, kwargs, to_csv=None):
    results = []
    initial_time = time.time()

    total_combinations = len(masked_sequences) * len(target_expressions)
    current_combination = 0
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    
    for element, masked_sequence in masked_sequences.items():
        for target_expression in target_expressions:
                if 'seed' in kwargs:
                    kwargs['seed'] += 1

                kwargs['masked_sequence'] = masked_sequence
        
                # Create genetic algorithm with the current parameter combination
                ga = CombinatorialAlgorithm(**kwargs, target_expression=target_expression, max_iter=1000)

                start = time.time()
                best_sequence, best_prediction, best_error = ga.run()
                end = time.time()

                # Store results
                result = {
                    'algorithm': 'Combinatorial Algorithm',
                    'element': element,
                    'masked_sequence': masked_sequence,
                    'mask_length': masked_sequence.count('N'),
                    'target_expression': target_expression,
                    'sequence': best_sequence,
                    'error': best_error,
                    'prediction': best_prediction,
                    'run_time': end - start
                }
                results.append(result)

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
    if to_csv != None:
        results_df.to_csv(to_csv, index=False)

    return results_df

def test_genetic(masked_sequences, target_expressions, lineages, kwargs, to_csv):
    results = []
    initial_time = time.time()

    total_combinations = len(masked_sequences) * len(target_expressions) * lineages
    current_combination = 0
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    
    for element, masked_sequence in masked_sequences.items():
        for target_expression in target_expressions:
            for lineage in range(lineages):
                if 'seed' in kwargs:
                    kwargs['seed'] += 1
                kwargs['masked_sequence'] = masked_sequence
                
                
                # Create genetic algorithm with the current parameter combination
                ga = GeneticAlgorithm(**kwargs, target_expression=target_expression)

                start = time.time()
                ga.run()
                end = time.time()

                # Store results
                result = {
                    'algorithm': 'Genetic Algorithm',
                    'element': element,
                    'masked_sequence': masked_sequence.count('N'), 
                    'mask_length': masked_sequence.count('N'),
                    'target_expression': target_expression,
                    'lineage': lineage,
                    'sequence': ga.best_sequence,
                    'error': abs(target_expression - ga.best_prediction),
                    'prediction': ga.best_prediction,
                    'run_time': end - start
                }
                results.append(result)

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
    if to_csv != None:
        results_df.to_csv(to_csv, index=False)

    return results_df

def combinatorial_scatter_plot(results_df, target_expression, index, color_column, color='tab10', polynomial_degree=1):
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    def add_best_fit_line(ax, x, y, degree, label, color):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color=color, linestyle='--')

    # Assign colors using colormap
    unique_colors = results_df[color_column].unique()
    color_map = plt.get_cmap(color)
    color_mapping = {name: color_map(i) for i, name in enumerate(unique_colors)}
    
    # Scatter Plot and Best-Fit Line for Error
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        axes[0].scatter(group[index], group['error'], alpha=0.7, color=scatter_color, label=name)
        add_best_fit_line(axes[0], group[index], group['error'], polynomial_degree, '', scatter_color)
    
    axes[0].set_xlabel(index)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Error vs {index} with Target Expression {target_expression}')
    axes[0].legend()

    # Scatter Plot and Best-Fit Line for Runtime
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        axes[1].scatter(group[index], group['run_time'], alpha=0.7, color=scatter_color, label=name)
        add_best_fit_line(axes[1], group[index], group['run_time'], polynomial_degree, '', scatter_color)
    
    axes[1].set_xlabel(index)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index} with Target Expression {target_expression}')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def combinatorial_scatter_plot_by_metric(results_df, metric, index, color_column, color='tab10', polynomial_degree=1):
    # Create the subplots, one for each target expression
    target_expressions = results_df['target_expression'].unique()
    fig, axes = plt.subplots(1, len(target_expressions), figsize=(7*len(target_expressions), 6))
    axes = axes.flatten()

    def add_best_fit_line(ax, x, y, degree, label, color):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color=color, linestyle='--')

    # Assign colors using colormap
    unique_colors = results_df[color_column].unique()
    color_map = plt.get_cmap(color)
    color_mapping = {name: color_map(i) for i, name in enumerate(unique_colors)}

    for i, target_expression in enumerate(target_expressions):
        for name, group in results_df.groupby(color_column):
            scatter_color = color_mapping[name]
            target_group = group[group['target_expression'] == target_expression]
            axes[i].scatter(target_group[index], target_group[metric], alpha=0.7, color=scatter_color, label=name)
            add_best_fit_line(axes[i], target_group[index], target_group[metric], polynomial_degree, '', scatter_color)
        
        axes[i].set_xlabel(index)
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Target Expression {target_expression}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

def combinatorial_scatter_plot_by_metric_element(results_df, metric, index, index_order, color_column, color='tab10', polynomial_degree=1):
    # Ensure the index column follows the specified index_order
    results_df[index] = pd.Categorical(results_df[index], categories=index_order, ordered=True)
    results_df = results_df.sort_values(by=index)
    
    # Create the subplots, one for each target expression
    target_expressions = results_df['target_expression'].unique()
    fig, axes = plt.subplots(1, len(target_expressions), figsize=(7 * len(target_expressions), 6))
    axes = axes.flatten() if len(target_expressions) > 1 else [axes]

    # Assign colors using colormap
    unique_colors = results_df[color_column].unique()
    color_map = plt.get_cmap(color)
    color_mapping = {name: color_map(i) for i, name in enumerate(unique_colors)}

    for i, target_expression in enumerate(target_expressions):
        for name, group in results_df.groupby(color_column):
            scatter_color = color_mapping[name]
            target_group = group[group['target_expression'] == target_expression]
            axes[i].scatter(target_group[index], target_group[metric], alpha=0.7, color=scatter_color, label=name)
        
        axes[i].set_xlabel(index)
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'Target Expression {target_expression}')
        axes[i].legend()
        axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels if necessary
    
    plt.tight_layout()
    plt.show()

def contribution_bar_graph(results_df, independent_variable, axhline=True, text=True, xlim=(0, 75), ylim=(0,1), order=None):

    pLac_expr = 0.33783603
    average_individual_results = results_df.groupby([independent_variable, 'target_expression'])['prediction'].mean().reset_index()

    # Sort the x axis by the independent variable, and set the order if specified
    if order:
        average_individual_results[independent_variable] = pd.Categorical(average_individual_results[independent_variable], categories=order, ordered=True)
    else:
        average_individual_results[independent_variable] = pd.Categorical(average_individual_results[independent_variable], ordered=True)

    average_individual_results = average_individual_results.sort_values(independent_variable)

    plt.figure(figsize=(10, 6))

    # Group by target expression
    data_target_0 = average_individual_results[average_individual_results['target_expression'] == 0]
    data_target_1 = average_individual_results[average_individual_results['target_expression'] == 1]

    # Plot the bar graph
    bars_0 = plt.bar(data_target_0[independent_variable], data_target_0['prediction'] - pLac_expr,
                     bottom=pLac_expr, label='Target Expression = 0', color='blue')
    bars_1 = plt.bar(data_target_1[independent_variable], data_target_1['prediction'] - pLac_expr,
                     bottom=pLac_expr, label='Target Expression = 1', color='orange')

    # Add text to the bars
    if text:
        for bar, label in zip(bars_0, data_target_0['prediction']):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y(),
                     f'{label:.2f}', ha='center', va='top', fontsize=10)

        for bar, label in zip(bars_1, data_target_1['prediction']):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y(),
                     f'{label:.2f}', ha='center', va='bottom', fontsize=10)

    # Add horizontal line for pLac expression
    if axhline:
        plt.axhline(y=pLac_expr, color='red', linestyle='--', label=f'pLac Expression = {pLac_expr:.2f}')

    # Add labels and title
    plt.xlabel(independent_variable)
    plt.ylabel('Relative Expression')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(f'Highest and Lowest Average Relative Expression for Each Masked pLac {independent_variable}')
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()

