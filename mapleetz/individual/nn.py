from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from mapleetz.individual.individual import Individual


def flatten_params(params: List[nn.Parameter]):
    with torch.no_grad():
        flattened = [p.data.view(-1) for p in params]
        return torch.cat(flattened, -1)


def apply_parameters(module: nn.Module, flattened_params: torch.Tensor):
    curr = 0
    with torch.no_grad():
        for p in module.parameters():
            if len(p.shape) >= 1:
                full_size = np.prod(p.shape)
                param_slice = flattened_params[curr : curr + full_size].view(p.shape)
                p.data = param_slice
                curr = curr + full_size


class TorchIndividual(Individual, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def params(self) -> torch.Tensor:
        return flatten_params(list(self.parameters()))

    @classmethod
    def create_from_params(
        cls, params: torch.Tensor, *args, **kwargs
    ) -> TorchIndividual:
        """
        Creates an individual from parameters

        Args:
            params: params

        Returns:
            Individual
        """
        individual = cls(*args, **kwargs)
        apply_parameters(individual, params)
        return individual
