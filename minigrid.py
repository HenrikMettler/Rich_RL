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
from functions import run_curriculum
from operators import Const05Node, Const2Node


def objective(
        individual: cgp.IndividualSingleGenome,
        network_params: dict,
        curriculum_params: dict,
        seeds
):

    if not individual.fitness_is_None():
        return individual
    try:
        individual.fitness = inner_objective(individual, network_params, curriculum_params, seeds)
    except ZeroDivisionError:
        individual.fitness = -np.inf
    return individual


# todo: test if decorator works properly!
@cgp.utils.disk_cache(
    "cache.pkl", compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
)
def inner_objective(
    ind: cgp.IndividualSingleGenome,
    network_params: dict,
    curriculum_params: dict,
    seeds
) -> float:

    rule = ind.to_torch()

    reward_per_seed = []
    reward_per_seed_mean = []
    for seed in seeds:
        seed = int(seed)

        torch.manual_seed(seed=seed)
        rng = np.random.default_rng(seed=seed)

        # environment and network initialization
        env = DynamicMiniGrid(seed=seed)
        env = ImgObsWrapper(env)
        state = env.respawn()["image"][:,:,0].flatten()

        policy_net = Network(n_inputs=np.size(state), **network_params)

        rewards_over_alterations = run_curriculum(env=env, net=policy_net, rule=rule, **curriculum_params, rng=rng)

        reward_per_seed.append(rewards_over_alterations)
        reward_per_seed_mean.append(np.mean(rewards_over_alterations))

    ind.reward_matrix = reward_per_seed
    reward_mean = np.mean(reward_per_seed_mean)

    return float(reward_mean)


def calculate_validation_fitness(champion, seed, network_params, curriculum_params):

    rule = champion.to_torch()

    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    # environment and network initialization
    env = DynamicMiniGrid(seed=seed)
    env = ImgObsWrapper(env)
    state = env.respawn()["image"][:, :, 0].flatten()

    policy_net = Network(n_inputs=np.size(state), **network_params)

    rewards_over_alterations = run_curriculum(env=env, net=policy_net, rule=rule, **curriculum_params, rng=rng)

    return rewards_over_alterations


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
    network_params = params['network_params']
    curriculum_params = params['curriculum_params']
    max_time = params['max_time']
    seeds = params["seeds"]
    genome_params = params['genome_params']  # {"n_inputs": 2}
    ea_params = params['ea_params']  # {'n_processes':4,}

    # initialize EA and Population
    ea = cgp.ea.MuPlusLambda(**ea_params)

    if params['use_drxeot_init']:
        pop = cgp.Population(genome_params=genome_params, individual_init=set_initial_dna)
    else:
        pop = cgp.Population(genome_params=genome_params)

    history = {}
    history["fitness_champion"] = []
    history["expression_champion"] = []
    history["reward_matrix"] = []

    champion_history = []   # not in history, since individuals can't be pickled

    def recording_callback(pop):
        history["fitness_champion"].append(pop.champion.fitness)
        history["expression_champion"].append(str(pop.champion.to_sympy()))
        history["reward_matrix"].append(pop.champion.reward_matrix)
        champion_history.append(pop.champion.copy())

    obj = functools.partial(objective, network_params=network_params, curriculum_params=curriculum_params,
                            seeds=seeds)

    start = time.time()
    cgp.evolve(obj,  max_time=max_time, ea=ea, pop=pop, print_progress=True, callback=recording_callback)
    end = time.time()

    history['validation_fitness'] = []
    history['validated_champion_expression'] = []
    history['validation_generation'] = []

    for generation, champion in enumerate(champion_history):
        if generation == 0 or history["fitness_champion"][generation] > history["fitness_champion"][generation-1]:
            seed = int(seeds[-1] +100)
            reward = calculate_validation_fitness(champion, seed, network_params, curriculum_params)
            history['validation_fitness'].append(reward)
            history['validated_champion_expression'].append(str(champion.to_sympy()))
            history['validation_generation'].append(generation)
    print(f"Time elapsed:", end - start)

    max_fitness = history["fitness_champion"][-1]
    best_expr = pop.champion.to_sympy()

    print(f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})')

    # store history
    filename = 'history.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(history, f)
