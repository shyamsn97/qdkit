from __future__ import annotations

from mapleetz.individual import Individual


class EvaluateOutput:
    def __init__(self, individual: Individual, states, fitness: float, **kwargs):
        self.individual = individual
        self.states = states
        self.fitness = fitness
