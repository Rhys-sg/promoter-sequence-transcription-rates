import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
import itertools
from scipy.stats import pearsonr, f_oneway, kruskal

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

    # Statistical Analysis
    def analyze_pivot_table(pivot_table, variable_name):
        # Flatten pivot table into group values
        grouped_values = []
        for column in pivot_table.columns:
            for row in pivot_table.index:
                grouped_values.append(pivot_table.loc[row, column])

        # Remove NaN values and group by index and columns
        grouped_values = [value for value in grouped_values if not np.isnan(value)]

        # Statistical test (e.g., ANOVA or Kruskal-Wallis)
        stat, p_value = kruskal(*grouped_values)  # Switch to f_oneway for ANOVA
        significance = "Significant" if p_value < 0.05 else "Not Significant"
        print(f"Statistical Test for {variable_name}:")
        print(f"  - Statistic: {stat:.3f}")
        print(f"  - p-value: {p_value:.3e}")
        print(f"  - {significance}")
        print()

    # Analyze Error
    analyze_pivot_table(error_pivot_table, 'error')

    # Analyze Run Time
    analyze_pivot_table(runtime_pivot_table, 'run_time')

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()

def analyze_relationship(x, y, name1, name2):
    r, p = pearsonr(x, y)
    significance = "Significant" if p < 0.05 else "Not Significant"
    print(f"Relationship between {name1} and {name2}:")
    print(f"  - Correlation Coefficient (r): {r:.3f}")
    print(f"  - p-value: {p:.3e}")
    print(f"  - {significance}")
    print()

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

    plt.tight_layout()
    plt.show()

    analyze_relationship(x1, y1, index, 'error')
    analyze_relationship(x2, y2, index, 'run_time')

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

    # Analyze the relationship between the variables and the metrics 
    analyze_relationship(results_df[index], results_df[metric1], index, metric1)
    analyze_relationship(results_df[index], results_df[metric2], index, metric2)


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
    def analyze_distribution(data, target, group_var, test_name="ANOVA"):
        grouped_data = [data[data[group_var] == level][target] for level in data[group_var].unique()]
        if test_name == "ANOVA":
            stat, p_value = f_oneway(*grouped_data)
        elif test_name == "Kruskal-Wallis":
            stat, p_value = kruskal(*grouped_data)
        else:
            raise ValueError("Unsupported test name. Use 'ANOVA' or 'Kruskal-Wallis'.")

        significance = "Significant" if p_value < 0.05 else "Not Significant"
        print(f"{test_name} Test for {target} by {group_var}:")
        print(f"  - Statistic: {stat:.3f}")
        print(f"  - p-value: {p_value:.3e}")
        print(f"  - {significance}")
        print()

    # Analyze the distributions for Error and Run Time
    analyze_distribution(results_df, 'error', index, test_name="ANOVA")
    analyze_distribution(results_df, 'run_time', index, test_name="ANOVA")

    # Adjust layout for better display
    plt.tight_layout()
    plt.show()