import numpy as np

from qdkit.individual import Individual


class EvaluateOutput:
    def __init__(self, individual: Individual, states, fitness: float, **kwargs):
        self.individual = individual
        self.states = states
        self.fitness = fitness


def gym_evaluate(individual: Individual, env):
    states = []
    state = env.reset()
    fitness = 0
    done = False
    states = [state]
    while not done:
        inp = state
        action = individual.act(inp)
        next_state, reward, done, _ = env.step(action)
        fitness += reward
        state = next_state
        states.append(state)
    return EvaluateOutput(states=state, fitness=fitness)


def normalize_arr(a: np.array, offset: float = 1e-5) -> np.array:
    normalized = ((a - a.min()) / (a.max() - a.min())) + offset
    return normalized / normalized.sum()


def sample_prob_index(p: np.array) -> np.array:
    i = np.random.choice(np.arange(p.size), p=p.ravel())
    return np.unravel_index(i, p.shape)


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
