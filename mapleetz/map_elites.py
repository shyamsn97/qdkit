import random
from dataclasses import dataclass  # noqa
from typing import Iterable, List, Union

from tqdm import trange

from mapleetz.evaluate_fn import EvaluateFunction, EvaluateOutput
from mapleetz.individual import Individual
from mapleetz.map import GridMap
from mapleetz.mutation import Mutation


@dataclass
class MapElitesConfig:
    mutations: List[Mutation]
    mutation_probs: List[float]


class MapElites:
    def __init__(
        self,
        initial_pop: List[Individual],
        map: GridMap,
        evaluate_fn: EvaluateFunction,
        mutations: Union[Iterable[Mutation], Mutation],
        choose_one_mutation: bool = False,
    ):
        self.initial_pop = initial_pop
        self.map = map
        self.evaluate_fn = evaluate_fn

        self.mutations = mutations
        if isinstance(mutations, Mutation):
            self.mutations = [mutations]

        self.choose_one_mutation = choose_one_mutation

    def mutate_individual(self, individual: Individual, it: int) -> Individual:
        mutations = self.mutations
        if self.choose_one_mutation:
            mutations = [random.choice(self.mutations)]
        for mutation in mutations:
            individual = mutation(individual, self.map, it)
        return individual

    def run(self, num_iterations: int):
        bar = trange(num_iterations)

        # initialize from population
        for individual in self.initial_pop:
            eval_output: EvaluateOutput = self.evaluate_fn(individual)

            self.map.add(
                individual,
                behavior_characteristic=eval_output.bc,
                fitness=eval_output.fitness,
            )

        for it in bar:
            individual = self.map.sample()
            mutated_individual = self.mutate_individual(individual, it)

            eval_output: EvaluateOutput = self.evaluate_fn(mutated_individual)

            self.map.add(
                mutated_individual,
                behavior_characteristic=eval_output.bc,
                fitness=eval_output.fitness,
            )
            bar.set_description(f"Max Fitness: {self.map.max_fitness()}")
