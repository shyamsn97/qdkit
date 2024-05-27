from __future__ import annotations

import random
from dataclasses import dataclass  # noqa
from typing import List, Optional, Tuple

import numpy as np

from qdkit.individual import Individual
from qdkit.utils import compute_ranks


class Niche:
    def __init__(self, niche_size: int, replace_criteria: str = "uniform"):
        self.niche_size = niche_size
        self.replace_criteria = replace_criteria
        self.individuals = []
        self.fitnesses = []

        self.elite_idx = None
        self.elite = None
        self.max_fitness = -float("inf")

    def sample_individual_idx(
        self, indices: Optional[List[int]] = None, sampling_method: str = "uniform"
    ) -> int:
        if indices is None:
            indices = list(range(len(self.individuals)))

        if sampling_method == "sorted":
            fitnesses = np.array([self.fitnesses[idx] for idx in indices])
            reverse_rankings = compute_ranks(fitnesses)[::-1] + 1e-5
            weights = reverse_rankings / np.sum(reverse_rankings)
            return random.choices(indices, weights=weights, k=1)[0]
        if sampling_method == "uniform":
            return random.choice(indices)

    def sample(self, sampling_method: str = "uniform"):
        sampled_index = self.sample_individual_idx(sampling_method=sampling_method)
        return self.individuals[sampled_index]

    def add(self, individual: Individual, fitness: float):
        if len(self.individuals) < self.niche_size:
            # if list is not full, we just append
            self.individuals.append(individual)
            self.fitnesses.append(fitness)
            replace_idx = self.__len__() - 1
        else:
            # if full, we have to replace an exisiting one
            indices = list(range(len(self.individuals)))
            indices.remove(self.elite_idx)  # ignore elite index
            replace_idx = self.sample_individual_idx(indices, self.replace_criteria)

        self.individuals[replace_idx] = individual
        self.fitnesses[replace_idx] = fitness

        if fitness >= self.max_fitness:
            self.elite_idx = replace_idx
            self.elite = self.individuals[replace_idx]
            self.max_fitness = fitness

    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]

    def __len__(self) -> int:
        return len(self.individuals)


@dataclass
class GridMapQueryOutput:
    niches: List[Niche]
    fitnesses: np.array
    occupancy: np.array
    indices: Tuple[np.array, ...]
