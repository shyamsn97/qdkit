from __future__ import annotations

import gym

from mapleetz.evaluate_fn.evaluate_fn import EvaluateOutput
from mapleetz.individual import Individual


def gym_evaluate(individual: Individual, env: gym.Env):
    states = []
    state = env.reset()
    fitness = 0
    done = False
    states = [state]
    while not done:
        inp = state
        action = individual.act(inp)
        next_state, reward, done, _ = env.step(action)
        fitness += reward
        state = next_state
        states.append(state)
    return EvaluateOutput(states=state, fitness=fitness)
