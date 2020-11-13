import numpy as np
import gym
import sympy
import functools
import torch

import cgp

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from network import Network


def inner_objective(f: Callable, network: Network, env: gym.Env, n_timesteps: int,
                    learning_rate: float, seed: int):

    env.seed(seed)

    cum_reward: float = 0.0
    observation: np.array = env.reset()

    for _ in range(n_timesteps):

        # compute forward pass and take a step
        hidden_activities, output_activities = network.forward(observation)
        action: np.ndarray = output_activities.detach().numpy()  # Todo: adapt for more than one output
        observation, reward, done, _ = env.step(action)

        # update the weights according to f
        network.update_weights(f=f, observation=observation, hidden_activities=hidden_activities,
                               output_activities=output_activities, learning_rate=learning_rate,
                               reward=reward)
        cum_reward += reward

        if done:
            observation = env.reset()

    env.close()

    return cum_reward


def objective(individual: cgp.IndividualSingleGenome, network: Network,
              env: gym.Env, n_timesteps, learning_rate: float, seed:int):
    if individual.fitness is not None:
        return individual

    f: Callable = individual.to_func()

    individual.fitness = inner_objective(f=f, network=network, env=env,
                                         n_timesteps=n_timesteps, learning_rate=learning_rate,
                                         seed=seed)

    return individual


seed = 1234
n_timesteps = 1000
learning_rate = 0.05

# population and evolutionary algorithm initialization
population_params = {"n_parents": 1, "mutation_rate": 0.03, "seed": seed}
genome_params = {
    "n_inputs": 4,  # pre, post, weight, reward
    "n_outputs": 1,
    "n_columns": 12,
    "n_rows": 1,
    "levels_back": 5,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}
ea_params = {"n_offsprings": 4, "tournament_size": 1, "n_processes": 1}
evolve_params = {"max_generations": 1000, "min_fitness": 10.0}

pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)

# environment initialization
env = gym.make("MountainCarContinuous-v0")

# network initialization
n_inputs = env.observation_space.shape[0]
n_hidden_layer = 10
n_outputs = env.action_space.shape[0]
network = Network(n_inputs=n_inputs, n_hidden_layer=n_hidden_layer, n_outputs=n_outputs)

# initialize a history
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


obj = functools.partial(objective, network=network, env=env, n_timesteps=n_timesteps,
                        learning_rate= learning_rate, seed=seed)
cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)