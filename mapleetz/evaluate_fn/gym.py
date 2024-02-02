from __future__ import annotations

from typing import Any, List, Tuple, Callable

import gym
import numpy as np

from mapleetz.evaluate_fn.evaluate_fn import EvaluateFunction, EvaluateOutput
from mapleetz.individual import Individual


class GymEvaluateFunction(EvaluateFunction):
    def __init__(self, env: gym.Env):
        self.env = env

    def behavior_space(self) -> List[Tuple[int, ...]]:
        return [
            tuple(self.env.observation_space.low),
            tuple(self.env.observation_space.high),
        ]

    def bc(self, states: List[np.array]) -> np.array:
        # by default use last state as bc
        return states[-1]

    def __call__(self, individual: Individual) -> EvaluateOutput:
        states = []
        state = self.env.reset()
        fitness = 0
        done = False
        states = [state]
        while not done:
            inp = state
            action = individual.act(inp)
            next_state, reward, done, _ = self.env.step(action)
            fitness += reward
            state = next_state
            states.append(state)
        bc = self.bc(states)
        return EvaluateOutput(fitness=fitness, bc=bc, aux={"states": states})
