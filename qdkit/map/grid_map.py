from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from qdkit.individual import Individual
from qdkit.map.map import Map
from qdkit.map.niche import GridMapQueryOutput, Niche
from qdkit.utils import EvaluateOutput, normalize_arr, sample_prob_index


class GridMap(Map):
    def __init__(
        self,
        behavior_characteristic_fn: Callable[[Individual, EvaluateOutput], Any],
        behavior_space: List[Tuple[int, ...]],
        n_bins: int,
        niche_size: int,
        niche_replace_criteria: str = "sorted",
        sampling_method: str = "sorted",
    ):
        self.behavior_characteristic_fn = behavior_characteristic_fn
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

    def bc(self, individual: Individual, eval_output: EvaluateOutput) -> Any:
        return self.behavior_characteristic_fn(individual, eval_output)

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
        bc = self.bc(individual=individual, eval_output=eval_output)
        bins, _ = self.query_individual(individual, bc)
        if self.niche_grid[bins] is None:
            self.niche_grid[bins] = Niche(self.niche_size, self.niche_replace_criteria)
        self.niche_grid[bins].add(individual, eval_output.fitness)
        self.fitness_grid[bins] = self.niche_grid[bins].max_fitness
        self.occupancy_grid[bins] = len(self.niche_grid[bins])

    def query_individual(
        self, individual: Individual, behavior_characteristic: np.array
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
