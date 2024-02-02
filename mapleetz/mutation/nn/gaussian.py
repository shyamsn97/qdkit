from typing import Optional

import numpy as np
import torch

from mapleetz.individual import Individual
from mapleetz.map import Map
from mapleetz.mutation.mutation import Mutation


class GaussianNoiseMutation(Mutation):
    def __init__(self, mu: float = 0.0, std: float = 1.0):
        self.mu = mu
        self.std = std

    def mutate(
        self,
        individual: Individual,
        map: Optional[Map] = None,
        it: Optional[int] = 0,
    ) -> Individual:
        with torch.no_grad():
            params = individual.params()
            noise = self.mu + np.random.randn(*tuple(params.shape)) * self.std

            if isinstance(params, torch.Tensor):
                noise = torch.from_numpy(noise).to(params.device)

            mutated_params = params + noise
            return individual.create_from_params(mutated_params)
