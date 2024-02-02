from __future__ import annotations

import random
from dataclasses import dataclass  # noqa
from typing import List, Optional, Tuple
import numpy as np

from mapleetz.individual import Individual
from mapleetz.map.map import Map
from mapleetz.evaluate_fn.evaluate_fn import EvaluateOutput
from mapleetz.utils import compute_ranks, normalize_arr, sample_prob_index


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


class GridMap(Map):
    def __init__(
        self,
        behavior_space: List[Tuple[int, ...]],
        n_bins: int,
        niche_size: int,
        niche_replace_criteria: str = "sorted",
        sampling_method: str = "sorted",
    ):
        self.behavior_space = behavior_space
        self.n_bins = n_bins
        self.bins = np.linspace(*self.behavior_space, n_bins + 1)[1:-1].T
        self.num_bcs = self.bins.shape[0]

        self.fitness_grid = np.zeros((self.n_bins,) * self.num_bcs) - 10**5
        self.niche_grid = np.empty((self.n_bins,) * self.num_bcs, dtype=Niche)
        self.occupancy_grid = np.zeros((self.n_bins,) * self.num_bcs, dtype=int)

        self.niche_size = niche_size
        self.niche_replace_criteria = niche_replace_criteria
        self.sampling_method = sampling_method

    def max_fitness(self) -> float:
        return np.max(self.fitness_grid)

    @property
    def num_individuals(self) -> int:
        return np.sum(self.occupancy_grid)

    def sample(self, sampling_method: Optional[str] = None) -> Individual:
        num_individuals = self.num_individuals

        if num_individuals == 0:
            return None

        if sampling_method is None:
            sampling_method = self.sampling_method

        if sampling_method == "uniform":
            probs = (self.occupancy_grid > 0).astype(float) / np.sum(
                (self.occupancy_grid > 0).astype(float)
            )
            sampled_index = sample_prob_index(probs)
        else:
            nonzero = (self.occupancy_grid > 0).astype(float)
            probs = (
                normalize_arr(self.fitness_grid) * nonzero
            )  # mask out stuff without individuals
            sampled_index = sample_prob_index(normalize_arr(probs, offset=0.0))

        return self.niche_grid[sampled_index].sample(sampling_method=sampling_method)

    def add(
        self,
        individual: Individual,
        eval_output: EvaluateOutput,
    ):
        bins, _ = self.query_individual(individual, eval_output.bc)
        if self.niche_grid[bins] is None:
            self.niche_grid[bins] = Niche(self.niche_size, self.niche_replace_criteria)
        self.niche_grid[bins].add(individual, eval_output.fitness)
        self.fitness_grid[bins] = self.niche_grid[bins].max_fitness
        self.occupancy_grid[bins] = len(self.niche_grid[bins])

    def query_individual(
        self, behavior_characteristic: np.array
    ) -> Tuple[Tuple[int, ...], Niche]:
        bins = [0] * behavior_characteristic.shape[0]
        for i in range(behavior_characteristic.shape[0]):
            bins[i] = np.digitize(behavior_characteristic[i], self.bins[i])
        bins = tuple(bins)
        return bins, self.niche_grid[bins]

    def search(self, fitness_criteria: float) -> GridMapQueryOutput:
        indices = np.where(self.fitness_grid > fitness_criteria)
        niches = self.niche_grid[indices]
        fitnesses = self.fitness_grid[indices]
        occupancy = self.occupancy_grid[indices]
        return GridMapQueryOutput(
            niches=niches, fitnesses=fitnesses, occupancy=occupancy, indices=indices
        )
