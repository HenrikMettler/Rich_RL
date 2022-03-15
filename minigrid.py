import numpy as np
import gym
import sympy
import functools
import pickle
import torch
import time

import cgp

from typing import List, Optional

from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid
from gym_minigrid.wrappers import ImgObsWrapper

from network import Network
from functions import play_episodes
from operators import Const05Node, Const2Node


def objective(
        individual: cgp.IndividualSingleGenome,
        prob_alteration_dict: dict,
        network_params: dict,
        env_params: dict,
):

    if not individual.fitness_is_None():
        return individual
    try:
        individual.fitness = inner_objective(individual, prob_alteration_dict, network_params, env_params)
    except ZeroDivisionError:
        individual.fitness = -np.inf
    return individual


@cgp.utils.disk_cache(
    "cache.pkl", compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
)
def inner_objective(
    ind: cgp.IndividualSingleGenome,
    prob_alteration_dict: dict,
    network_params: dict,
    env_params: dict,
) -> float:

    t = ind.to_torch()

    seeds = env_params["seeds"]

    max_n_alterations = env_params["max_n_alterations"]
    n_alterations_per_new_env = env_params["n_alterations_per_new_env"]
    n_episodes_per_alteration = env_params["n_episodes_per_alteration"]
    n_steps_max = env_params["n_steps_max"]
    temporal_novelty_decay = env_params["temporal_novelty_decay"]
    spatial_novelty_time_decay = env_params["spatial_novelty_time_decay"]
    spatial_novelty_distance_decay = env_params["spatial_novelty_distance_decay"]

    reward_per_seed_mean = []
    for seed in seeds:
        seed = int(seed)

        torch.manual_seed(seed=seed)
        rng = np.random.default_rng(seed=seed)

        # environment and network initialization
        env = DynamicMiniGrid(seed=seed)
        env = ImgObsWrapper(env)
        state = env.respawn()["image"].flatten()

        policy_net = Network(n_inputs=np.size(state), **network_params)

        rewards_over_alterations: List[float] = []

        for n_alter in range(1, max_n_alterations):

            # environement altering
            env.alter_env(n=n_alterations_per_new_env, prob_alteration_dict=prob_alteration_dict,
                            spatial_novelty_distance_decay=spatial_novelty_distance_decay)

            # runs
            rewards_over_episodes = play_episodes(env=env, net=policy_net, rule=t,
                                                  n_episodes=n_episodes_per_alteration, n_steps_max=n_steps_max,
                                                  temporal_novelty_decay=temporal_novelty_decay,
                                                  spatial_novelty_time_decay=spatial_novelty_time_decay,
                                                  rng=rng)

            rewards_over_alterations.append(np.mean(rewards_over_episodes))
            env.respawn()

        reward_per_seed_mean.append(np.mean(rewards_over_alterations))
    reward_mean = np.mean(reward_per_seed_mean)
    return float(reward_mean)


def set_initial_dna(ind):
    genome = cgp.Genome(**genome_params)
    genome.randomize(rng=np.random.RandomState(seed=1234))

    dna_prior = [2, 0, 1]  # Mul as 3rd operator (2), r as first (0), el as second (1) input
    genome.set_expression_for_output(dna_insert=dna_prior)
    ind = cgp.IndividualSingleGenome(genome)
    ind.to_sympy()

    return cgp.IndividualSingleGenome(genome)


if __name__ == "__main__":

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)

    # parse params
    prob_alteration_dict = params['prob_alteration_dict']
    network_params = params['network_params']
    env_params = params['env_params']
    max_time = params['max_time']
    genome_params = params['genome_params']  # {"n_inputs": 2}
    ea_params = params['ea_params']  # {'n_processes':4,}

    # initialize EA and Population
    ea = cgp.ea.MuPlusLambda(**ea_params)

    if params['use_rxet_init']:
        pop = cgp.Population(genome_params=genome_params, individual_init=set_initial_dna)
    else:
        pop = cgp.Population(genome_params=genome_params)

    # Todo: writing the history like this is not good, sympy expressions can not be pickled (i think)
    history = {}
    history["fitness_champion"] = []
    history["expression_champion"] = []
    def recording_callback(pop):
        history["fitness_champion"].append(pop.champion.fitness)
        history["expression_champion"].append(pop.champion.to_sympy())

    obj = functools.partial(objective, prob_alteration_dict=prob_alteration_dict,
                            network_params=network_params, env_params=env_params)

    start = time.time()
    cgp.evolve(obj,  max_time=max_time, ea=ea, pop=pop, print_progress=True, callback=recording_callback)
    end = time.time()
    print(f"Time elapsed:", end - start)

    max_fitness = history["fitness_champion"][-1]
    best_expr = pop.champion.to_sympy()

    print(f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})')

    # store history
    filename = 'history.pickle'
    file = open(filename, 'wb')
    pickle.dump(history, file)
