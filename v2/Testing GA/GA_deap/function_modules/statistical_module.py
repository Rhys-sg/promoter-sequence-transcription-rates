import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway, kruskal

def analyze_relationship(x, y, name1, name2):

    # Ensure numeric data
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Optimal values
    averaged_y = y.groupby(x).mean()
    optimal_idx = averaged_y.idxmin()
    print(f"Optimal {name1} for {name2}: {optimal_idx} (Minimum {name2}: {averaged_y[optimal_idx]:.3f})")

    # Pearson correlation
    r, p = pearsonr(x.dropna(), y.dropna())
    print(f"Relationship between {name1} and {name2}:")
    print(f"  - Correlation Coefficient (r): {r:.3f}")
    print(f"  - p-value: {p:.3e}")
    print(f"  - {'Significant' if p < 0.05 else 'Not Significant'}")
    print()

def analyze_distribution(data, target, group_var):
    # Ensure the target column is numeric
    data = data.copy() 
    data[target] = pd.to_numeric(data[target], errors='coerce')

    # Find the index with the lowest average value
    avg_stats = data.groupby(group_var)[target].mean()
    optimal_idx = avg_stats.idxmin()
    print(f"Optimal {group_var} for {target}: {optimal_idx} (Average {target}: {avg_stats[optimal_idx]:.3f})")

    # ANOVA Test
    grouped_data = [data[data[group_var] == level][target] for level in data[group_var].unique()]
    f, p = f_oneway(*grouped_data)
    print(f"ANOVA Test for {target} by {group_var}:")
    print(f"  - F-statistic: {f:.3f}")
    print(f"  - p-value: {p:.3e}")
    print(f"  - {'Significant' if p < 0.05 else 'Not Significant'}")
    print()

def analyze_pivot_table(pivot_table, variable_name):
    # Flatten pivot table into group values
    grouped_values = []
    for column in pivot_table.columns:
        for row in pivot_table.index:
            grouped_values.append(pivot_table.loc[row, column])

    # Remove NaN values and group by index and columns
    grouped_values = [value for value in grouped_values if not np.isnan(value)]

    # Find combination with lowest average value
    optimal_idx = np.argmin(grouped_values)
    optimal_row = pivot_table.index[optimal_idx // len(pivot_table.columns)]
    optimal_column = pivot_table.columns[optimal_idx % len(pivot_table.columns)]
    row_name = pivot_table.index.name
    column_name = pivot_table.columns.name

    # Print optimal combination
    print(f"Optimal {row_name} for {variable_name}: {optimal_column:.3f}")
    print(f"Optimal {column_name} for {variable_name}: {optimal_row:.3f}")
    print(f"  - Average {variable_name}: {grouped_values[optimal_idx]:.3f}")
    print()

    # Statistical test (e.g., ANOVA or Kruskal-Wallis)
    stat, p_value = kruskal(*grouped_values)  # Switch to f_oneway for ANOVA
    significance = "Significant" if p_value < 0.05 else "Not Significant"
    print(f"Statistical Test for {variable_name}:")
    print(f"  - Statistic: {stat:.3f}")
    print(f"  - p-value: {p_value:.3e}")
    print(f"  - {significance}")
    print()