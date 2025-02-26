import pandas as pd
import math
import random
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def reconstruct_sequence(infill, masked_sequence, mask_indices):
    sequence = list(masked_sequence)
    for idx, char in zip(mask_indices, infill):
        sequence[idx] = char
    return sequence

def generate_nucleotide():
    nucleotide = [0, 0, 0, 0]
    nucleotide[random.randint(0, 3)] = 1
    return tuple(nucleotide)

def generate_individual(mask_indices):
    return [generate_nucleotide() for _ in range(len(mask_indices))]

def generate_population(n, masked_sequence, mask_indices):
    return [reconstruct_sequence(generate_individual(mask_indices), masked_sequence, mask_indices) for _ in range(n)]

def append_pca(df):
    X = np.stack(df['sequence'].values)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    return pd.concat([df, df_pca], axis=1)

def plot_PCA_contour(df, x_points=100, y_points=100):
    X = df['PC1'].values
    Y = df['PC2'].values
    Z = df['prediction'].values

    grid_x, grid_y = np.meshgrid(
        np.linspace(X.min(), X.max(), x_points),
        np.linspace(Y.min(), Y.max(), y_points)
    )
    grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')
    levels = np.linspace(Z.min(), Z.max(), 12)

    plt.style.use('_mpl-gallery-nogrid')

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap='coolwarm')
    cbar = plt.colorbar(contour)
    cbar.set_label("Prediction Value")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Contour Map of PCA-transformed Data")

    plt.show()
