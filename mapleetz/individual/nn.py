from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from functorch import make_functional, make_functional_with_buffers

from mapleetz.individual.individual import Individual
from mapleetz.utils import count_torch_params


class FunctionalParamVectorWrapper(nn.Module):
    """
    This wraps a module so that it takes params in the forward pass
    """

    def __init__(self, module: nn.Module):
        super(FunctionalParamVectorWrapper, self).__init__()
        self.custom_buffers = None
        param_dict = dict(module.named_parameters())
        self.target_weight_shapes = {k: param_dict[k].size() for k in param_dict}
        self.num_parameters = count_torch_params(module)

        try:
            _functional, self.named_params = make_functional(module)
        except Exception:
            _functional, self.named_params, buffers = make_functional_with_buffers(
                module
            )
            self.custom_buffers = buffers
        self.functional = [_functional]  # remove params from being counted

    def forward(self, param_vector: torch.Tensor, *args, **kwargs):
        params = []
        start = 0
        for p in self.named_params:
            end = start + np.prod(p.size())
            params.append(param_vector[start:end].view(p.size()))
            start = end
        if self.custom_buffers is not None:
            return self.functional[0](params, self.custom_buffers, *args, **kwargs)
        return self.functional[0](params, *args, **kwargs)


class TorchModuleIndividual(Individual, nn.Module):
    def __init__(self, func_model: FunctionalParamVectorWrapper, params: torch.Tensor):
        super().__init__()
        self.func_model = func_model
        self.nn_params = nn.Parameter(params.float())
        self.num_target_parameters = params.shape[0]

        self.__device_param_dummy__ = nn.Parameter(
            torch.empty(0)
        )  # to keep track of device

    @property
    def device(self) -> torch.device:
        return self.__device_param_dummy__.device

    def params(self) -> torch.Tensor:
        return self.nn_params.data

    @classmethod
    def initialize_params(cls, params_shape: Iterable[int]):
        return torch.randn(params_shape).float()

    def create_from_params(self, params: torch.Tensor) -> TorchModuleIndividual:
        return TorchModuleIndividual(self.func_model, params)

    @classmethod
    def from_target(cls, target_module: nn.Module) -> Individual:
        func_model = FunctionalParamVectorWrapper(target_module)
        params = cls.initialize_params(func_model.num_parameters)
        return TorchModuleIndividual(func_model, params)

    def copy(self) -> TorchModuleIndividual:
        func_model = self.func_model
        params = self.params()
        return self.__class__(func_model=func_model, params=params)

    def forward(
        self, *args, params: torch.Tensor = None, requires_grad: bool = False, **kwargs
    ):
        if params is None:
            params = self.params()
        if requires_grad:
            func = self.func_model.forward
        else:
            func = torch.no_grad()(self.func_model.forward)
        return func(params, *args, **kwargs)