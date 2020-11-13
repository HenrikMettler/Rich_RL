import numpy as np
import gym
import sympy
import functools

import cgp

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set


def inner_objective(f: Callable, seed: int):
    raise NotImplementedError


def objective(individual: cgp.IndividualSingleGenome, seed: int):
    if individual.fitness is not None:
        return individual

    f = individual.to_func()

    individual.fitness = inner_objective(f, seed)

    return individual


seed = 1234
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
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


obj = functools.partial(objective, seed=seed)
cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)