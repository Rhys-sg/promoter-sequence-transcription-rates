import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import itertools

from GA.TestGeneticAlgorithm import GeneticAlgorithm
from .statistical_module import analyze_relationship, analyze_distribution, analyze_pivot_table

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
                        'sequence': ga.best_sequences[0],
                        'error': abs(target_expression - ga.best_predictions[0]),
                        'prediction': ga.best_predictions[0],
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

def test_param_convergence(param_ranges, target_expressions, lineages, kwargs, to_csv=None, iteration=1):
    results_dfs = {}
    initial_time = time.time()
    
    # Generate all combinations of parameters
    param_keys = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    param_combinations = list(itertools.product(*param_values))

    total_combinations = len(param_combinations) * len(target_expressions)
    current_combination = 0
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    
    for param_combination in param_combinations:
        params = dict(zip(param_keys, param_combination))
        
        for target_expression in target_expressions:
            try:
                if 'seed' in kwargs:
                    kwargs['seed'] += 1
                
                kwargs = {**kwargs, **params}
                
                # Create genetic algorithm with the current parameter combination
                ga = GeneticAlgorithm(**kwargs, target_expression=target_expression, track_history=True)
                ga.run(lineages)
                
                # Store results
                results_dfs[tuple(param_combination)] = pd.DataFrame({
                    '''
                    THIS DOES NOT WORK, storing both arrays and floats cause issues with convergence_fill_between
                    Fix before graphing contour plots
                    '''
                    # # Contains lineage history for each generation, shape: (lineages, generations)
                    # 'population_history' : ga.reorder_history_by_generation(ga.population_history),
                    # 'best_sequence_history' : ga.reorder_history_by_generation(ga.best_sequence_history),
                    # 'best_fitness_history' : ga.reorder_history_by_generation(ga.best_fitness_history),
                    # 'best_prediction_history' : ga.reorder_history_by_generation(ga.best_prediction_history),
                    # 'convergence_history' : ga.reorder_history_by_generation(ga.convergence_history),

                    # Contains lineage statistics for each generation, shape: (generations,)
                    'min_convergence_history' : ga.min_lineage_convergence_history,
                    'max_convergence_history' : ga.max_lineage_convergence_history,
                    'mean_convergence_history' : ga.mean_lineage_convergence_history,
                }).T

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
                print(f'Params: {params}, Target Expression: {target_expression}')
                continue
    
    # Close progress bar
    progress_bar.close()
    
    # Save to CSV
    if to_csv != None:
        for params, df in results_dfs.items():
            df.to_csv(f'{to_csv}_convergence/{params}', index=False)

    return results_dfs


def format_time(time_in_seconds):
    if time_in_seconds < 60:
        return f'{time_in_seconds:.2f}s'
    if time_in_seconds < 3600:
        return f'{time_in_seconds / 60:.2f}min'
    return f'{time_in_seconds / 3600:.2f}h'

def heatmap(results_df, target_expression, index, columns, figsize=(14, 6)):
    # Create pivot tables for error and runtime
    error_pivot_table = results_df.pivot_table(values='error', index=index, columns=columns, aggfunc='mean')
    runtime_pivot_table = results_df.pivot_table(values='run_time', index=index, columns=columns, aggfunc='mean')
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap for Mean Error
    sns.heatmap(error_pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=axes[0])
    axes[0].set_title(f'Mean Error for {index} and {columns} with Target Expression {target_expression}')
    axes[0].set_xlabel(columns)
    axes[0].set_ylabel(index)
    axes[0].set_xticklabels([f'{round(float(tick.get_text()), 2)}' for tick in axes[0].get_xticklabels()], rotation=0)
    axes[0].set_yticklabels([tick.get_text() for tick in axes[0].get_yticklabels()], rotation=0)

    # Heatmap for Run Time
    sns.heatmap(runtime_pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=axes[1])
    axes[1].set_title(f'Run Time for {index} and {columns} with Target Expression {target_expression}')
    axes[1].set_xlabel(columns)
    axes[1].set_ylabel(index)
    axes[1].set_xticklabels([f'{round(float(tick.get_text()), 2)}' for tick in axes[1].get_xticklabels()], rotation=0)
    axes[1].set_yticklabels([tick.get_text() for tick in axes[1].get_yticklabels()], rotation=0)

    # Analyze Error
    analyze_pivot_table(error_pivot_table, 'error')
    analyze_pivot_table(runtime_pivot_table, 'run_time')

    plt.tight_layout()
    plt.show()

def scatter_plot(results_df, target_expression, index, polynomial_degree=1):
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    def add_best_fit_line(ax, x, y, degree, label):
        # Fit a polynomial to the data
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color='red', linestyle='--')

    # Error vs index
    x1 = results_df[index]
    y1 = results_df['error']
    axes[0].scatter(x1, y1, alpha=0.7)
    add_best_fit_line(axes[0], x1, y1, polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[0].set_xlabel(index)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Error vs {index} with Target Expression {target_expression}')
    axes[0].legend()

    # Runtime vs index
    x2 = results_df[index]
    y2 = results_df['run_time']
    axes[1].scatter(x2, y2, alpha=0.7)
    add_best_fit_line(axes[1], x2, y2, polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[1].set_xlabel(index)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index} with Target Expression {target_expression}')
    axes[1].legend()

    # Statistical Analysis
    analyze_relationship(x1, y1, index, 'error')
    analyze_relationship(x2, y2, index, 'run_time')

    plt.tight_layout()
    plt.show()

def scatter_plot_overlaid_separate(results_df, target_expression, index, color_column, color='tab10', metric1='error', metric2='run_time', polynomial_degree=1):
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
    
    # Scatter Plot and Best-Fit Line for metric1
    plt.figure(figsize=(7, 6))
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        plt.scatter(group[index], group[metric1], alpha=0.7, color=scatter_color, label=f'{color_column}={name}')
        add_best_fit_line(plt, group[index], group[metric1], polynomial_degree, '', scatter_color)
    plt.xlabel(index)
    plt.ylabel(metric1)
    plt.title(f'{metric1} vs {index} with Target Expression {target_expression}')
    plt.legend()
    plt.show()

    # Scatter Plot and Best-Fit Line for metric2
    plt.figure(figsize=(7, 6))
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        plt.scatter(group[index], group[metric2], alpha=0.7, color=scatter_color, label=f'{color_column}={name}')
        add_best_fit_line(plt, group[index], group[metric2], polynomial_degree, '', scatter_color)
    plt.xlabel(index)
    plt.ylabel(metric2)
    plt.title(f'{metric2} vs {index} with Target Expression {target_expression}')
    plt.legend()
    plt.show()

    # Statistical Analysis
    analyze_relationship(results_df[index], results_df[metric1], index, metric1)
    analyze_relationship(results_df[index], results_df[metric2], index, metric2)


def scatter_plot_overlaid(results_df, target_expression, index, color_column, color='tab10', metric1='error', metric2='run_time', polynomial_degree=1):
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
    
    # Scatter Plot and Best-Fit Line for metric1
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        axes[0].scatter(group[index], group[metric1], alpha=0.7, color=scatter_color, label=f'{color_column}={name}')
        add_best_fit_line(axes[0], group[index], group[metric1], polynomial_degree, '', scatter_color)
    
    axes[0].set_xlabel(index)
    axes[0].set_ylabel(metric1)
    axes[0].set_title(f'{metric1} vs {index} with Target Expression {target_expression}')
    axes[0].legend()

    # Scatter Plot and Best-Fit Line for metric2
    for name, group in results_df.groupby(color_column):
        scatter_color = color_mapping[name]
        axes[1].scatter(group[index], group[metric2], alpha=0.7, color=scatter_color, label=f'{color_column}={name}')
        add_best_fit_line(axes[1], group[index], group[metric2], polynomial_degree, '', scatter_color)
    
    axes[1].set_xlabel(index)
    axes[1].set_ylabel(metric2)
    axes[1].set_title(f'{metric2} vs {index} with Target Expression {target_expression}')
    axes[1].legend()

    # Statistical Analysis
    analyze_relationship(results_df[index], results_df[metric1], index, metric1)
    analyze_relationship(results_df[index], results_df[metric2], index, metric2)

    plt.tight_layout()
    plt.show()

def distribution_plot(results_df, target_expression, index, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distribution of Error
    sns.kdeplot(data=results_df, x='error', hue=index, fill=False, ax=axes[0])
    axes[0].set_title(f'Distribution of Error with Target Expression {target_expression}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')

    # Distribution of Run Time
    sns.kdeplot(data=results_df, x='run_time', hue=index, fill=False, ax=axes[1])
    axes[1].set_title(f'Distribution of Run Time with Target Expression {target_expression}')
    axes[1].set_xlabel('Run Time (s)')
    axes[1].set_ylabel('Frequency')

    # Statistical Analysis
    analyze_distribution(results_df, 'error', index)
    analyze_distribution(results_df, 'run_time', index)

    plt.tight_layout()
    plt.show()

def convergence_plot(results_df, figsize=(14, 6), label='Lineage'):
    '''
    Plot the hamming distance convergence for each lineage.
    Takes in a dataframe of results with rows of lineages and columns of generations.
    '''
    fig, ax = plt.subplots(figsize=figsize)

    for lineage in results_df.index:
        ax.plot(results_df.columns, results_df.loc[lineage], label=f'{label} {lineage}')

    ax.set_xlabel('Generation')
    ax.set_ylabel('Hamming Distance')
    ax.set_title('Hamming Distance Convergence')
    ax.legend()
    plt.show()

def convergence_fill_between(results_dfs, color='tab10', figsize=(14, 6)):
    '''
    Plot the hamming distance convergence using fill_between to show min, average, and max.
    Takes in a dataframe of results with rows of lineages and columns of generations.
    '''
    fig, ax = plt.subplots(figsize=figsize)

    # Assign colors using colormap
    color_map = plt.get_cmap(color)
    color_mapping = {name: color_map(i) for i, name in enumerate(results_dfs.keys())}

    # Plot the convergence history for dataframe
    for params, df in results_dfs.items():
        generations = df.columns
        ax.fill_between(generations, df.loc['min_convergence_history'], df.loc['max_convergence_history'], alpha=0.3)
        ax.plot(generations, df.loc['mean_convergence_history'], color=color_mapping[params], linestyle='-', linewidth=2, label=params)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Hamming Distance')
    ax.set_title('Hamming Distance Convergence')
    ax.legend()
    
    plt.show()