import random
from dataclasses import dataclass  # noqa
from typing import Callable, Iterable, List, Union

from tqdm import trange

from miniqd.individual import Individual
from miniqd.map import GridMap
from miniqd.mutation import Mutation
from miniqd.utils import EvaluateOutput


@dataclass
class MapElitesConfig:
    mutations: List[Mutation]
    mutation_probs: List[float]


class MapElites:
    def __init__(
        self,
        initial_pop: List[Individual],
        map: GridMap,
        evaluate_fn: Callable[[Individual], EvaluateOutput],
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
        mutated_individual = individual.clone()
        if self.choose_one_mutation:
            mutations = [random.choice(self.mutations)]
        for mutation in mutations:
            mutated_individual = mutation(mutated_individual, self.map, it)
        return individual

    def run(self, num_iterations: int):
        bar = trange(num_iterations)

        # initialize from population
        for individual in self.initial_pop:
            eval_output: EvaluateOutput = self.evaluate_fn(individual)

            self.map.add(
                individual,
                eval_output=eval_output,
            )

        for it in bar:
            individual = self.map.sample()
            mutated_individual = self.mutate_individual(individual, it)
            eval_output: EvaluateOutput = self.evaluate_fn(mutated_individual)

            self.map.add(
                individual=mutated_individual,
                eval_output=eval_output,
            )
            bar.set_description(f"Max Fitness: {self.map.max_fitness()}")
