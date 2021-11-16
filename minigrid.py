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
from functions import alter_env, update_weights, initialize_genome_with_rxet_prior
from operators import Const05Node, Const2Node


def objective(
        individual: cgp.IndividualSingleGenome,
        prob_alteration_dict: dict,
        ):

    if not individual.fitness_is_None():
        return individual
    try:
        individual.fitness = inner_objective(individual, prob_alteration_dict)
    except ZeroDivisionError:
        individual.fitness = -np.inf
    return individual


def inner_objective(
    ind: cgp.IndividualSingleGenome,
    prob_alteration_dict: dict
) -> float:

    t = ind.to_torch()

    seeds = np.linspace(1234567890, 1234567899, 9)

    weight_update_mode = 'evolved-rule'

    max_n_alterations = 10
    n_episodes_per_alteration: int = 2000
    n_steps_max: int = 100

    reward_per_seed_mean = []
    for seed in seeds:
        seed = int(seed)

        torch.manual_seed(seed=seed)
        rng = np.random.default_rng(seed=seed)

        # general environment parametrisation
        env = DynamicMiniGrid(seed=seed)
        env = ImgObsWrapper(env)
        state = env.respawn()["image"].flatten()

        # network parameterization
        n_inputs: int = np.size(state)
        n_hidden: int = 30
        n_outputs: int = 3  # Left, right, forward (pick up, drop, toggle, done are ingnored); env.action_space.n
        learning_rate: float = 0.02

        policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                             learning_rate=learning_rate, weight_update_mode=weight_update_mode)

        n_steps_per_episode: List[int] = []
        rewards_over_episodes: List[float] = []

        for n_alter in range(1, max_n_alterations):

            # environement altering
            env = alter_env(env=env, n=1, prob_alteration_dict=prob_alteration_dict)

            # runs
            for episode in range(n_episodes_per_alteration):
                state = env.respawn()["image"].flatten()
                log_probs: List[torch.Tensor] = []
                probs: List[float] = []
                actions: List[int] = []
                hidden_activities_all = []
                rewards: List[float] = []

                for steps in range(n_steps_max):

                    action, prob, hidden_activities = policy_net.get_action(state, rng)

                    hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
                    log_prob = torch.log(prob.squeeze(0)[action])

                    new_state, reward, done, _ = env.step(action)
                    log_probs.append(log_prob)
                    probs.append(prob)
                    actions.append(action)
                    rewards.append(reward)
                    hidden_activities_all.append(hidden_activities)

                    if done or steps == n_steps_max-1:
                        update_params = {
                            "rewards": rewards,
                            "probs": probs,
                            "log_probs": log_probs,
                            "actions": actions,
                            "hidden_activities": hidden_activities_all,
                        }
                        update_weights(network=policy_net, **update_params, weight_update_mode=weight_update_mode,
                                       normalize_discounted_rewards_b=False, rule=t)

                        n_steps_per_episode.append(steps)
                        break

                    state = new_state.flatten()
                rewards_over_episodes.append(sum(rewards))

            env.respawn()
        reward_per_seed_mean.append(np.mean(rewards_over_episodes))
    reward_mean = np.mean(reward_per_seed_mean)
    return float(reward_mean)


def set_initial_dna(ind):
    genome = cgp.Genome()
    genome.dna = initialize_genome_with_rxet_prior(n_inputs=2,
                                                   n_hidden=128,  # from default in library
                                                   n_operators=4,  # from default in library
                                                   max_arity=2,
                                                   rng=np.random.default_rng(seed=1234)
                                                   )

    return cgp.IndividualSingleGenome(genome)


if __name__ == "__main__":

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)

    prob_alteration_dict = params['prob_alteration_dict']
    evolve_params = {"max_objective_calls": 100} # 84600s ≃  23h 30 min
    genome_params ={"n_inputs": 2}

    if params['use_rxet_init']:
        pop = cgp.Population(genome_params=genome_params, individual_init=set_initial_dna)
    else:
        pop = cgp.Population(genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda()

    history = {}
    history["fitness_champion"] = []
    history["expression_champion"] = []

    def recording_callback(pop):
        history["fitness_champion"].append(pop.champion.fitness)
        history["expression_champion"].append(pop.champion.to_sympy())

    obj = functools.partial(objective, prob_alteration_dict=prob_alteration_dict)

    start = time.time()
    cgp.evolve(obj, pop, ea, **evolve_params, print_progress=True, callback=recording_callback, )
    end = time.time()
    print(f"Time elapsed:", end - start)

    max_fitness = history["fitness_champion"][-1]
    best_expr = pop.champion.to_sympy()

    print(f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})')

    # store history
    filename = 'history.pickle'
    file = open(filename, 'wb')
    pickle.dump(history, file)