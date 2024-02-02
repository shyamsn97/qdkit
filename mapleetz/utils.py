import numpy as np
import torch.nn as nn


def normalize_arr(a: np.array, offset: float = 1e-5) -> np.array:
    normalized = ((a - a.min()) / (a.max() - a.min())) + offset
    return normalized / normalized.sum()


def sample_prob_index(p: np.array) -> np.array:
    i = np.random.choice(np.arange(p.size), p=p.ravel())
    return np.unravel_index(i, p.shape)


def count_torch_params(module: nn.Module):
    return sum([np.prod(p.size()) for p in module.parameters()])


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= x.size - 1
    y -= 0.5
    return y
