import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm

from GA_params_class.GeneticAlgorithm import GeneticAlgorithm

def test_params(param_range, param_name, cnn_model_path, masked_sequence, target_expressions, precision, verbose, lineages, iteration=1, seed=1):
    results = []
    total_combinations = len(target_expressions) * len(param_range)
    progress_bar = tqdm(total=total_combinations, desc='Processing combinations', position=0)
    initial_time = time.time()

    for target_expression in target_expressions:
        for i, param_val in enumerate(param_range):
            # Dynamically set the dependent parameter using kwargs
            ga_kwargs = {
                param_name: param_val  # Add the parameter dynamically
            }
            ga = GeneticAlgorithm(
                cnn_model_path=cnn_model_path,
                masked_sequence=masked_sequence,
                target_expression=target_expression,
                precision=precision,
                verbose=verbose,
                seed=seed,
                **ga_kwargs  # Pass dynamically created kwargs
            )
            # Time the run
            start_time = time.time()
            best_sequences, best_predictions = ga.run(lineages)
            end_time = time.time()

            # Record the results
            for sequence, prediction in zip(best_sequences, best_predictions):
                results.append({
                    'target_expression': target_expression,
                    param_name: param_val,
                    'sequence': sequence,
                    'error': abs(prediction - target_expression),
                    'run_time': (end_time - start_time) / lineages
                })
                
            # Update progress bar
            progress_bar.update(1)
            elapsed_time = time.time() - initial_time
            eta = ((elapsed_time / (i+1)) * (total_combinations - (i+1)))
            if eta > 60:
                eta_message = f'{eta/60:.2f}min'
            else:
                eta_message = f'{eta:.2f}s'
            progress_bar.set_postfix({
                'Elapsed': f'{elapsed_time:.2f}s',
                'ETA': eta_message
            })

    # Close progress bar
    progress_bar.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'Data/individual_params/{param_name}_results_{iteration}.csv', index=False)

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
    sns.kdeplot(data=results_df, x='error', hue='selection', fill=False, ax=axes[0])
    axes[0].set_title(f'Distribution of Error with Target Expression {target_expression}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')

    # Distribution of Run Time
    sns.kdeplot(data=results_df, x='error', hue='selection', fill=False, ax=axes[1])
    axes[1].set_title(f'Distribution of Run Time with Target Expression {target_expression}')
    axes[1].set_xlabel('Run Time (s)')
    axes[1].set_ylabel('Frequency')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def box_plot(results_df, target_expression, index, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Distribution of Error
    sns.boxplot(data=results_df, x='selection', y='error', palette='Set2', ax=axes[0])
    axes[0].set_title(f'Distribution of Error with Target Expression {target_expression}')
    axes[0].set_xlabel('Error')
    axes[0].set_ylabel('Frequency')

    # Distribution of Run Time
    sns.boxplot(data=results_df, x='selection', y='run_time', palette='Set2', ax=axes[1])
    axes[1].set_title(f'Distribution of Run Time with Target Expression {target_expression}')
    axes[1].set_xlabel('Run Time (s)')
    axes[1].set_ylabel('Frequency')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show() 