import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import itertools

from GA.GeneticAlgorithm import GeneticAlgorithm

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
    
    # Close progress bar
    progress_bar.close()

    results_df = pd.DataFrame(results)
    if to_csv != None:
        results_df.to_csv(to_csv, index=False)

    return results_df

def bayesian_test(params, cnn_model_path, masked_sequence, target_expressions, precision, verbose, lineages, seed=1):
    print(f'Testing params: {params}', end='')
    error = 0
    run_time = 0

    for target_expression in target_expressions:
        ga = GeneticAlgorithm(
            cnn_model_path=cnn_model_path,
            masked_sequence=masked_sequence,
            target_expression=target_expression,
            precision=precision,
            verbose=verbose,
            seed=seed,
            **params
        )
        # Time the run
        start_time = time.time()
        _, best_predictions = ga.run(lineages)
        end_time = time.time()

        # Record the results
        error += sum([abs(prediction - target_expression) for prediction in best_predictions]) / lineages
        run_time += (end_time - start_time) / lineages
    error /= len(target_expressions)
    print(f' - Error: {error}, Run Time: {run_time}')

    return error, run_time

def format_time(time_in_seconds):
    if time_in_seconds < 60:
        return f'{time_in_seconds:.2f}s'
    if time_in_seconds < 3600:
        return f'{time_in_seconds / 60:.2f}min'
    return f'{time_in_seconds / 3600:.2f}h'

def heatmap(results_df, target_expression, index, columns, figsize=(14, 6)):
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

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def scatter_plot(results_df, target_expression, index1, index2=None, polynomial_degree=1):
    # Create the subplots
    num_rows = 1 if index2 is None else 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
    axes = axes.flatten()

    def add_best_fit_line(ax, x, y, degree, label):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color='red', linestyle='--')

    # Error vs index1
    axes[0].scatter(results_df[index1], results_df['error'], alpha=0.7)
    add_best_fit_line(axes[0], results_df[index1], results_df['error'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[0].set_xlabel(index1)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Error vs {index1} with Target Expression {target_expression}')
    axes[0].legend()

    # Runtime vs index1
    axes[1].scatter(results_df[index1], results_df['run_time'], alpha=0.7)
    add_best_fit_line(axes[1], results_df[index1], results_df['run_time'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[1].set_xlabel(index1)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index1} with Target Expression {target_expression}')
    axes[1].legend()

    if index2 is None:
        plt.tight_layout()
        plt.show()
        return

    # Error vs index2
    axes[2].scatter(results_df[index2], results_df['error'], alpha=0.7)
    add_best_fit_line(axes[2], results_df[index2], results_df['error'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[2].set_xlabel(index2)
    axes[2].set_ylabel('Error')
    axes[2].set_title(f'Error vs {index2} with Target Expression {target_expression}')
    axes[2].legend()

    # Runtime vs index2
    axes[3].scatter(results_df[index2], results_df['run_time'], alpha=0.7)
    add_best_fit_line(axes[3], results_df[index2], results_df['run_time'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[3].set_xlabel(index2)
    axes[3].set_ylabel('Runtime (s)')
    axes[3].set_title(f'Runtime vs {index2} with Target Expression {target_expression}')
    axes[3].legend()

    plt.tight_layout()
    plt.show()

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

    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()

def hex_bin(results_df, target_expression, index1, index2=None, polynomial_degree=1):
    # Create the subplots
    num_rows = 1 if index2 is None else 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
    axes = axes.flatten()

    def add_best_fit_line(ax, x, y, degree, label):
        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        x_vals = np.linspace(min(x), max(x), 500)
        y_vals = poly(x_vals)
        ax.plot(x_vals, y_vals, label=label, color='red', linestyle='--')

    # Error vs index1
    hb = axes[0].hexbin(results_df[index1], results_df['error'], gridsize=30, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=axes[1], label='Count')
    add_best_fit_line(axes[0], results_df[index1], results_df['error'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[0].set_xlabel(index1)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Error vs {index1} with Target Expression {target_expression}')
    axes[0].legend()

    # Runtime vs index1
    hb = axes[0].hexbin(results_df[index1], results_df['run_time'], gridsize=30, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=axes[1], label='Count')
    add_best_fit_line(axes[1], results_df[index1], results_df['run_time'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[1].set_xlabel(index1)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index1} with Target Expression {target_expression}')
    axes[1].legend()

    if index2 is None:
        plt.tight_layout()
        plt.show()
        return

    # Error vs index2
    hb = axes[2].hexbin(results_df[index2], results_df['error'], gridsize=30, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=axes[1], label='Count')
    add_best_fit_line(axes[2], results_df[index2], results_df['error'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[2].set_xlabel(index2)
    axes[2].set_ylabel('Error')
    axes[2].set_title(f'Error vs {index2} with Target Expression {target_expression}')
    axes[2].legend()

    # Runtime vs index2
    hb = axes[3].hexbin(results_df[index2], results_df['run_time'], gridsize=30, cmap='Blues', mincnt=1)
    fig.colorbar(hb, ax=axes[1], label='Count')
    add_best_fit_line(axes[3], results_df[index2], results_df['run_time'], polynomial_degree, f'{polynomial_degree}-Degree Fit')
    axes[3].set_xlabel(index2)
    axes[3].set_ylabel('Runtime (s)')
    axes[3].set_title(f'Runtime vs {index2} with Target Expression {target_expression}')
    axes[3].legend()

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

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def box_plot(results_df, target_expression, index, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distribution of Error
    sns.boxplot(data=results_df, x=index, y='error', palette='Set2', ax=axes[0])
    axes[0].set_title(f'Distribution of Error with Target Expression {target_expression}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')

    # Distribution of Run Time
    sns.boxplot(data=results_df, x=index, y='run_time', palette='Set2', ax=axes[1])
    axes[1].set_title(f'Distribution of Run Time with Target Expression {target_expression}')
    axes[1].set_xlabel('Run Time (s)')
    axes[1].set_ylabel('Frequency')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show() 