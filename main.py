import numpy as np
import gym
import sympy
import functools

import cgp

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set


def inner_objective(f: Callable, env: gym.Env, n_timesteps: int, seed: int):

    env.seed(seed)

    cum_reward: float = 0.0
    observation = env.reset()

    for _ in range(n_timesteps):

        # Todo: adapt what the observation is, what is feed into
        continuous_action = f(observation)
        observation, reward, done, _ = env.step(continuous_action)
        cum_reward += reward

        if done:
            observation = env.reset()

    env.close()

    return cum_reward


def objective(individual: cgp.IndividualSingleGenome, env: gym.Env, n_timesteps, seed:int):
    if individual.fitness is not None:
        return individual

    f: Callable = individual.to_func()

    individual.fitness = inner_objective(f=f, env=env, n_timesteps=n_timesteps, seed=seed)

    return individual


seed = 1234

# cgp parameterisation
population_params = {"n_parents": 1, "mutation_rate": 0.03, "seed": seed}
genome_params = {
    "n_inputs": 4,  # pre, post, weight, reward
    "n_outputs": 1,
    "n_columns": 12,
    "n_rows": 1,
    "levels_back": 5,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}
ea_params = {"n_offsprings": 4, "tournament_size": 2, "n_processes": 2}
evolve_params = {"max_generations": 1000, "min_fitness": 0.0}

pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)

# environment parameterisation
env = gym.make('Mountain')

# initialize a history
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


obj = functools.partial(objective, seed=seed)
cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)