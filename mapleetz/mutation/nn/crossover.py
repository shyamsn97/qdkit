from typing import List, Optional, Tuple

import numpy as np
import torch

from mapleetz.individual import Individual
from mapleetz.map import Map
from mapleetz.mutation.mutation import Mutation


class CrossoverMutation(Mutation):
    def __init__(
        self,
        parameter_proportion: float = 0.1,
        sampling_method: str = "uniform",
        sample_segment: bool = True,
    ):
        self.parameter_proportion = parameter_proportion
        self.sampling_method = sampling_method
        self.sample_segment = sample_segment

    def _get_segment_range(self, individual: Individual) -> Tuple[List[int], int]:
        flattened_params = individual.params.flatten()
        num_parameters = flattened_params.shape[0]
        segment_num_params = int(num_parameters * self.parameter_proportion)
        segment_min_range = 0
        segment_max_range = num_parameters - segment_num_params
        segment_idx = np.random.randint(segment_min_range, segment_max_range, 1)[0]
        return (
            list(range(segment_idx, segment_idx + segment_num_params)),
            segment_num_params,
        )

    def _sample_param_indices(self, individual: Individual) -> Tuple[List[int], int]:
        if self.sample_segment:
            return self._get_segment_range(individual)

        flattened_params = individual.params.flatten()
        num_parameters = flattened_params.shape[0]
        segment_num_params = int(num_parameters * self.parameter_proportion)
        indices = list(range(num_parameters))
        segment_indices = list(
            np.random.choice(indices, size=segment_num_params, replace=False)
        )
        return segment_indices, segment_num_params

    def mutate(
        self,
        individual: Individual,
        map: Map,
        it: Optional[int] = 0,
    ) -> Individual:
        with torch.no_grad():
            crossover_individual = map.sample(sampling_method=self.sampling_method)
            params = individual.params
            flattened_params = params.flatten()
            crossover_individual_flattened_params = (
                crossover_individual.params.flatten()
            )

            mutated_params = params

            num_parameters = flattened_params.shape[0]

            segment_indices, segment_num_params = self._get_segment_range(individual)
            zeros = np.zeros((segment_num_params,))
            ones = np.ones((num_parameters,))

            if isinstance(flattened_params, torch.Tensor):
                zeros = torch.from_numpy(zeros).to(flattened_params.device)
                ones = torch.from_numpy(ones).to(flattened_params.device)

            crossover_segment = (
                zeros + crossover_individual_flattened_params[segment_indices]
            )

            # mask out segment
            ones[segment_indices] *= zeros

            # mask out individual segment and replace with crossover_individual segment
            mutated_params = flattened_params * ones
            mutated_params[segment_indices] += crossover_segment

            # reshape to original params
            mutated_params = mutated_params.reshape(tuple(individual.params.shape))

        return individual.create_from_params(mutated_params)
