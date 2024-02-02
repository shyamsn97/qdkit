from __future__ import annotations

import abc
from abc import ABCMeta
from typing import Any, Union

import numpy as np
import torch


class Individual(metaclass=ABCMeta):
    @abc.abstractmethod
    def params(self) -> Union[np.array, torch.Tensor, Any]:
        """
        Returns a param vector for an individual. This could either be a numpy array or a torch tensor

        Returns:
            Union[np.array, torch.Tensor]
        """
