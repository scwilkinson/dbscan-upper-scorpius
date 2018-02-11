import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from .settings import MIN_PTS, FEATURES, DEFAULT_EPS


def _get_k_distance(df):

    feature_df = df[FEATURES].copy()
    feature_df[FEATURES] = StandardScaler().fit_transform(feature_df)

    k_neighbours_matrix = kneighbors_graph(feature_df.as_matrix(), MIN_PTS, mode='distance').toarray()

    sorted_k_distance_vector = np.flipud(np.sort(np.amax(k_neighbours_matrix, axis=1)))

    return sorted_k_distance_vector


def _plot_k_distance(vector):

    plt.grid(which='major')

    plt.xlabel("$\mathrm{point}$")
    plt.ylabel("$\mathrm{" + str(MIN_PTS) + "-distance}$")

    plt.xlim([0, len(vector) - 1])
    plt.ylim([0, 2])  # Set this to a sensible value!

    plt.scatter(np.arange(len(vector)), vector)


def choose_eps(df):

    sorted_k_distance_vector = _get_k_distance(df)

    choice = input("Show k-distance plot? [y/n] ")

    if choice == 'y':
        _plot_k_distance(sorted_k_distance_vector)
        plt.show()

    eps = input("Choose an eps (Default = {}): ".format(DEFAULT_EPS))

    if not eps:
        eps = DEFAULT_EPS

    return float(eps)
