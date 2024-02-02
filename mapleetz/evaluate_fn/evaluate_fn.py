from __future__ import annotations

import abc
from abc import ABCMeta
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from mapleetz.individual import Individual


@dataclass
class EvaluateOutput:
    fitness: float
    bc: np.array
    aux: List[Dict[str, Any]] = field(default_factory=lambda: [{}])

    def __add__(self, other):
        if isinstance(other, EvaluateOutput):
            fitness = self.fitness + other.fitness
            bc = self.bc + other.bc
            aux = self.aux + other.aux
            return EvaluateOutput(fitness=fitness, bc=bc, aux=aux)
        else:
            fitness = self.fitness + other
            bc = self.bc + bc
            return EvaluateOutput(fitness=fitness, bc=bc, aux=self.aux)


class EvaluateFunction(metaclass=ABCMeta):
    @abc.abstractmethod
    def behavior_space(self, *args, **kwargs) -> List[Tuple[int, ...]]:
        """
        return behavior space

        Returns:
            List[Tuple[int, ...]]
        """

    @abc.abstractmethod
    def __call__(self, individual: Individual, *args, **kwargs) -> EvaluateOutput:
        """
        Returns a dataclass with fitness and bc

        Returns:
            EvaluateOutput
        """
