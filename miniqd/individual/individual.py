from __future__ import annotations

import abc
from abc import ABCMeta
from typing import Any


class Individual(metaclass=ABCMeta):
    def clone(self) -> Individual:
        return self.create_from_params(self.params)

    @property
    @abc.abstractmethod
    def params(self) -> Any:
        """
        Returns a param vector for an individual. This could either be a numpy array or a torch tensor

        Returns:
            Union[np.array, torch.Tensor]
        """

    @classmethod
    @abc.abstractmethod
    def create_from_params(cls, params: Any, *args, **kwargs) -> Individual:
        """
        Creates an individual from parameters

        Args:
            params: params

        Returns:
            Individual
        """
