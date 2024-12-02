import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time

from GA_params_test import GeneticAlgorithm

def test_combination(cnn_model_path, masked_sequence, target_expression, precision, run_per_combination, **kwargs):
    errors = []
    run_times = []
    for run_id in range(run_per_combination):
        ga = GeneticAlgorithm(
            cnn_model_path=cnn_model_path,
            masked_sequence=masked_sequence,
            target_expression=target_expression,
            precision=precision,
            print_progress=False,
            **kwargs
        )
        # Time the run
        start_time = time.time()
        best_sequence, best_prediction = ga.run()
        end_time = time.time()

        # Record the results
        errors.append(abs(best_prediction - target_expression))
        run_times.append(end_time - start_time)

    return np.mean(errors), np.mean(run_times)

def heatmap(results_df, index, columns):
    error_pivot_table = results_df.pivot_table(values='error', index=index, columns=columns, aggfunc='mean')
    runtime_pivot_table = results_df.pivot_table(values='run_time', index=index, columns=columns, aggfunc='mean')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Heatmap for Mean Error
    sns.heatmap(error_pivot_table, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
    axes[0].set_title(f'Mean Error for {index} and {columns}')
    axes[0].set_xlabel(columns)
    axes[0].set_ylabel(index)

    # Heatmap for Run Time
    sns.heatmap(runtime_pivot_table, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])
    axes[1].set_title(f'Run Time for {index} and {columns}')
    axes[0].set_xlabel(columns)
    axes[0].set_ylabel(index)

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def scatter_plot(results_df, index, polynomial_degree):
    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Error vs Number of Parents
    axes[0].scatter(results_df[index], results_df['error'], label='Error Data', alpha=0.7)
    z_error_parents = np.polyfit(results_df[index], results_df['error'], polynomial_degree)
    p_error_parents = np.poly1d(z_error_parents)
    axes[0].plot(results_df[index], p_error_parents(results_df[index]), color='red', linestyle='--', label='Quadratic Best Fit')
    axes[0].set_xlabel(index)
    axes[0].set_ylabel('Error')
    axes[0].set_title(f'Error vs {index}')
    axes[0].legend()

    # Runtime vs Number of Parents
    axes[1].scatter(results_df[index], results_df['run_time'], label='Runtime Data', alpha=0.7)
    z_runtime_parents = np.polyfit(results_df[index], results_df['run_time'], polynomial_degree)
    p_runtime_parents = np.poly1d(z_runtime_parents)
    axes[1].plot(results_df[index], p_runtime_parents(results_df[index]), color='blue', linestyle='--', label='Quadratic Best Fit')
    axes[1].set_xlabel(index)
    axes[1].set_ylabel('Runtime (s)')
    axes[1].set_title(f'Runtime vs {index}')
    axes[1].legend()