import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time

def heatmap(results_df, target_expression, index, columns, figsize=(14, 6)):
    error_pivot_table = results_df.pivot_table(values='error', index=index, columns=columns, aggfunc='mean')
    runtime_pivot_table = results_df.pivot_table(values='run_time', index=index, columns=columns, aggfunc='mean')
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap for Mean Error
    sns.heatmap(error_pivot_table, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
    axes[0].set_title(f'Mean Error for {index} and {columns} with Target Expression {target_expression}')
    axes[0].set_xlabel(columns)
    axes[0].set_ylabel(index)
    axes[0].set_xticklabels([f'{round(float(tick.get_text()), 2)}' for tick in axes[0].get_xticklabels()], rotation=0)
    axes[0].set_yticklabels([tick.get_text() for tick in axes[0].get_yticklabels()], rotation=0)

    # Heatmap for Run Time
    sns.heatmap(runtime_pivot_table, annot=True, fmt=".2f", cmap="viridis", ax=axes[1])
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