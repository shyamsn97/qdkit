import abc
from abc import ABCMeta
from typing import List, Optional

import numpy as np

from miniqd.individual.individual import Individual
from miniqd.map.map import Map


class Mutation(metaclass=ABCMeta):
    @abc.abstractmethod
    def mutate(
        self,
        individual: Individual,
        map: Optional[Map],
        it: Optional[int],
        *args,
        **kwargs,
    ) -> Individual:
        """
        Mutate function that creates a new individual!

        Args:
            individual (Individual): individual to be mutated
            map (GridMap): Optional for mutation use
            it (int): Optional parameter denoting current itteration in the map elites training cycle

        Returns:
            Individual: new individual
        """

    def __call__(
        self,
        individual: Individual,
        map: Optional[Map],
        it: Optional[int],
        *args,
        **kwargs,
    ) -> Individual:
        return self.mutate(individual, map, *args, **kwargs)


class MutationSet(Mutation):
    """
    MutationSet: Used to combine multiple mutations into one!.

    Args:
        mutations (List[Mutation]) : list of mutations
        mutation_probs (List[float]): Optional list of mutation probabilities that are used to sample mutations at each step.
        By default these are set to one, meaning all mutations will be applied on each step
    """

    def __init__(
        self, mutations: List[Mutation], mutation_probs: Optional[List[float]] = None
    ):
        self.mutations = mutations
        self.mutation_probs = mutation_probs
        if mutation_probs is None:
            self.mutation_probs = np.ones(len(self.mutations))
        assert len(self.mutation_probs) == len(self.mutations)

    def mutate(
        self,
        individual: Individual,
        map: Optional[Map] = None,
        it: Optional[int] = 0,
    ) -> Individual:
        new_individual = individual.clone()
        for mutation, mutation_prob in zip(self.mutations, self.mutation_probs):
            if np.random.uniform() < mutation_prob:
                new_individual = mutation(new_individual, map, it)
        return new_individual
