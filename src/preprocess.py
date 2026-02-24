import numpy as np
import copy

def laplacian_filter(X, channels, neighbours):

    laplacian = copy.deepcopy(X)

    for channel, n in zip(channels, neighbours):
        laplacian[:, channel, :] = X[:, channel, :] - np.mean(X[:, n, :], axis=1)

    return laplacian


def channel_aggregation(X):

    aggregated = copy.deepcopy(X)

    aggregated = np.mean(aggregated, axis=1, keepdims=True)

    return aggregated


