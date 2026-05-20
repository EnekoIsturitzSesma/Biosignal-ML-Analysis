import numpy as np
import scipy.linalg
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


def euclidean_alignment(X):
    covs = np.array([x @ x.T / x.shape[-1] for x in X])
    R_mean = covs.mean(axis=0)
    
    R_inv_sqrt = np.linalg.inv(scipy.linalg.sqrtm(R_mean)).real
    
    return np.array([R_inv_sqrt @ x for x in X])


def apply_ea_loso(X, subjects):
    X_aligned = X.copy()
    for subj in np.unique(subjects):
        mask = subjects == subj
        X_aligned[mask] = euclidean_alignment(X[mask])
    return X_aligned


def apply_ea_loso_multiband(X, subjects):
    X_aligned = X.copy()
    for b in range(X.shape[1]):
        X_aligned[:, b, :, :] = apply_ea_loso(X[:, b, :, :], subjects)
    return X_aligned

