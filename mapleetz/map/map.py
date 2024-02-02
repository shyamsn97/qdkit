import abc
from abc import ABCMeta

from mapleetz.evaluate_fn.evaluate_fn import EvaluateOutput
from mapleetz.individual.individual import Individual


class Map(metaclass=ABCMeta):

    @property
    @abc.abstractmethod
    def num_individuals(self) -> int:
        """
        Returns:
            int: Number of individuals in the map
        """

    @abc.abstractmethod
    def sample(self, sampling_method: str = "uniform") -> Individual:
        """Samples an individual from the map

        Returns:
            Individual: sampled individual
        """

    @abc.abstractmethod
    def add(self, individual: Individual, eval_output: EvaluateOutput):
        """Adds an individual to the map

        Args:
            eval_output (EvaluateOutput): output from evaluate function
        """
