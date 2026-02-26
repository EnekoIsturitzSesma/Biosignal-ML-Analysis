import numpy as np
import copy

def laplacian_filter(X, channels, neighbours, use_multiband=False):

    laplacian = copy.deepcopy(X)

    if use_multiband:
        n_bands = X.shape[1]
        for b in range(n_bands):
            for channel, n in zip(channels, neighbours):
                laplacian[:, b, channel, :] = X[:, b, channel, :] - np.mean(X[:, b, n, :], axis=1)
    else:
        for channel, n in zip(channels, neighbours):
            laplacian[:, channel, :] = X[:, channel, :] - np.mean(X[:, n, :], axis=1)

    return laplacian


def channel_aggregation(X, use_multiband=False):

    aggregated = copy.deepcopy(X)

    if use_multiband:
        n_bands = X.shape[1]
        for b in range(n_bands):
            aggregated = np.mean(X, axis=2, keepdims=True)
    else:
        aggregated = np.mean(aggregated, axis=1, keepdims=True)

    return aggregated


def normalize_trial(X):

    if X.ndim == 3:
        mean = np.mean(X, axis=2, keepdims=True)
        std = np.std(X, axis=2, keepdims=True)
    elif X.ndim == 2:
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)

    normalized = (X - mean) / (std + 1e-8)

    return normalized

