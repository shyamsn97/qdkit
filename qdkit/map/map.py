import abc
from abc import ABCMeta

from qdkit.individual.individual import Individual
from qdkit.map.niche import GridMapQueryOutput
from qdkit.utils import EvaluateOutput


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

    @abc.abstractmethod
    def search(self, fitness_criteria: float) -> GridMapQueryOutput:
        """Search for individuals that reach a fitness criteria

        Args:
            fitness_criteria (float): _description_

        Returns:
            GridMapQueryOutput: _description_
        """
