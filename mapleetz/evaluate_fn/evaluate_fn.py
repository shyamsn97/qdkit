from __future__ import annotations

import abc
from abc import ABCMeta
from typing import  List, Tuple, Any

import numpy as np

from mapleetz.individual import Individual


class EvaluateOutput:
    def __init__(
        self,
        states,
        fitness: float,
        bc: np.array,
        **kwargs
    ):
        self.states = states
        self.fitness = fitness
        self.bc = bc


class EvaluateFunction(metaclass=ABCMeta):

    @abc.abstractmethod
    def behavior_space(self) -> List[Tuple[int, ...]]:
        """
        behavior space

        Returns:
            List[Tuple[int, ...]]: _description_
        """


    @abc.abstractmethod
    def bc(self, states: Any) -> Any:
        """
        Behavior characteristic function

        Returns:
            Any: a behavior characteristic
        """

    @abc.abstractmethod
    def __call__(self, individual: Individual, *args, **kwargs) -> EvaluateOutput:
        """
        Returns a dataclass with fitness and bc

        Returns:
            EvaluateOutput
        """
