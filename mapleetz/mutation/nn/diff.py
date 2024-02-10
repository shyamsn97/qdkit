from typing import Optional

import numpy as np
import torch

from mapleetz.individual import Individual
from mapleetz.map import Map
from mapleetz.mutation.mutation import Mutation


class DiffMutation(Mutation):
    def __init__(self, sigma_1: float = 0.1, sigma_2: float = 0.25):
        # from https://rl-vs.github.io/rlvs2021/class-material/evolutionary/light-virtual_school_qd.pdf
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

    def mutate(
        self,
        individual: Individual,
        map: Map,
        it: Optional[int] = 0,
    ) -> Individual:
        with torch.no_grad():
            individual_2 = map.sample(sampling_method=self.sampling_method)

            params = individual.params
            params_2 = individual_2.params

            noise = np.random.randn(*tuple(params.shape))
            noise_2 = np.random.randn(*tuple(params.shape))

            if isinstance(params, torch.Tensor):
                noise = torch.from_numpy(noise).to(params.device)
                noise_2 = torch.from_numpy(noise_2).to(params.device)

            # xi_t+1 = xi_t + sigma*N(0,1) + sigma_2*(xj_t - xi_t)*N(0,1)
            mutated_params = (
                params
                + noise * self.sigma_1
                + self.sigma_2 * ((params_2 - params) * noise_2)
            )

        return individual.create_from_params(mutated_params)
