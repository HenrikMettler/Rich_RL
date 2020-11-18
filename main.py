import numpy as np
import gym
import torch
import sympy
import functools
import warnings
import time
import tracemalloc

import cgp

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from network import Network


def inner_objective(
    t: torch.nn.Module,
    network: Network,
    env: gym.Env,
    n_runs_per_individual: int,
    n_steps_per_run: int,
    learning_rate: float,
    seed: int,
) -> float:

    env.seed(seed)

    cum_reward_all_episodes: List = []
    for _ in range(n_runs_per_individual):
        observation: np.array = env.reset()
        cum_reward_this_episode: float = 0

        for _ in range(n_steps_per_run):
            # compute forward pass and take a step
            hidden_activities, output_activities = network.forward(observation)
            action: np.ndarray = output_activities.detach().numpy()  # Todo: adapt for more than one output
            if np.isnan(action):  # return early if actions diverge
                return -np.inf

            observation, reward, done, _ = env.step(action)

            # update the weights according to f
            network.update_weights(
                t=t,
                observation=observation,
                hidden_activities=hidden_activities,
                output_activities=output_activities,
                reward=reward,
                learning_rate=learning_rate,
            )
            cum_reward_this_episode += reward

            if done:
                cum_reward_all_episodes.append(cum_reward_this_episode)
                cum_reward_this_episode = 0
                observation = env.reset()

    env.close()
    cum_reward: float = np.mean(cum_reward_all_episodes)

    return cum_reward


def objective(
    individual: cgp.IndividualSingleGenome,
    env: gym.Env,
    n_runs_per_individual: int,
    n_steps_per_run: int,
    learning_rate: float,
    seed: int,
):
    if individual.fitness is not None:
        return individual

    # network initialization
    torch.manual_seed(seed=seed)
    n_inputs = env.observation_space.shape[0]
    n_hidden_layer = 100
    n_outputs = env.action_space.shape[0]
    network = Network(
        n_inputs=n_inputs, n_hidden_layer=n_hidden_layer, n_outputs=n_outputs
    )

    t = individual.to_torch()
    try:
        with warnings.catch_warnings():  # ignore warnings due to zero division
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in double_scalars"
            )
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in double_scalars"
            )
            individual.fitness = inner_objective(
                t=t,
                network=network,
                env=env,
                n_runs_per_individual=n_runs_per_individual,
                n_steps_per_run=n_steps_per_run,
                learning_rate=learning_rate,
                seed=seed,
            )
    except ZeroDivisionError:
        individual.fitness = -np.inf

    return individual


seed = 1000
n_runs_per_individual = 3
n_steps_per_run = 1000
learning_rate = 0.05

# population and evolutionary algorithm initialization
population_params = {"n_parents": 1, "mutation_rate": 0.03, "seed": seed}
genome_params = {
    "n_inputs": 4,  # pre, post, weight, reward
    "n_outputs": 1,
    "n_columns": 25,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}
ea_params = {"n_offsprings": 4, "tournament_size": 1, "n_processes": 1}
evolve_params = {"max_generations": 1000, "min_fitness": n_runs_per_individual * 90}
# MountainCarContinuous v0 solved with reward 90
# (https://github.com/openai/gym/wiki/MountainCarContinuous-v0)

pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)

# environment initialization
env = gym.make("MountainCarContinuous-v0")

# initialize a history
history = {}
history["fitness_champion"] = []
# history["expression_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)
    # history["expression_champion"].append(pop.champion.to_sympy())


obj = functools.partial(
    objective,
    env=env,
    n_runs_per_individual=n_runs_per_individual,
    n_steps_per_run=n_steps_per_run,
    learning_rate=learning_rate,
    seed=seed,
)


start = time.time()
cgp.evolve(
    pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback
)
end = time.time()
print(f"Time elapsed:", end - start)

max_fitness = history["fitness_champion"][-1]
best_expr = pop.champion.to_sympy()
# best_expr = best_expr.replace("x_0", "pre").replace("x_1", "post").replace("x_2, "weight")
# .replace("x_4", "reward")
print(
    f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})  '
    f"for {n_runs_per_individual} timesteps per evaluation"
)
