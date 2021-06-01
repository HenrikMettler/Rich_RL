import numpy as np
import gym
import torch
import sympy
import functools
import time
import pickle

from typing import Optional, AnyStr

import cgp

from network import Network
from functions import update_el_traces, update_weights_online_with_rule, initialize_genome_with_rxet_prior
from operators import Const05Node, Const2Node

using_pickled_params = True  # option for pre written parameters written in write_job

if using_pickled_params:
    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)

    gamma = params['gamma']
    use_online_init = params['use_online_init']
    seed = params['seed']
    n_episodes = params['n_episodes']
    cum_reward_threshold = params['cum_reward_threshold']
    population_params = params['population_params']
    ea_params = params['ea_params']
    evolve_params = params['evolve_params']
    genome_params = params['genome_params']

else:
    gamma = 0.9
    use_online_init = True
    seed = 12345
    n_episodes = 100
    cum_reward_threshold = 250000
    n_episodes_reward_expectation = 100
    population_params = {"n_parents": 1, "seed": seed}
    ea_params = {"n_offsprings": 4, "mutation_rate": 0.03, "reorder_genome": True, "n_processes": 1,
                 "hurdle_percentile": [0.5, 0.0], }
    evolve_params = {"max_generations": 2}

    genome_params = {
        "n_inputs": 4,  # reward, el_traces, done (episode termination), expected_cum_reward_episode
        "n_outputs": 1,
        "n_columns": 300,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (cgp.Mul, cgp.Add, cgp.Sub, cgp.ConstantFloat, Const05Node, Const2Node)
    }


def set_initial_dna(ind):
    genome = cgp.Genome(**genome_params)
    genome.dna = initialize_genome_with_rxet_prior(n_inputs=genome_params["n_inputs"],
                                                   n_hidden=genome_params["n_columns"],
                                                   n_operators=len(genome_params["primitives"]),
                                                   max_arity=2,
                                                   rng= np.random.default_rng(seed=1234) # Todo: is using a fixed seed here an issue?
                                                   )

    return cgp.IndividualSingleGenome(genome)


def inner_objective(
    t: torch.nn.Module,
    network: Network,
    env: gym.Env,
    seed: int,
    rng: np.random.Generator,
    mode: AnyStr,
    gamma: Optional[float] = 0.9,
    n_steps_per_run: Optional[int] = 200,
    n_episodes: Optional[int] = 0,
    cum_reward_threshold: Optional[int] = 0,
    n_episodes_reward_expectation: Optional[float] = 100,

) -> float:

    env.seed(seed)
    cum_reward = 0
    episode_counter = 0
    expected_cum_reward_per_episode = 0
    while episode_counter < n_episodes or cum_reward <= cum_reward_threshold:
        state = env.reset()
        el_traces = torch.zeros([network.output_layer.out_features, network.output_layer.in_features + 1])
        discounted_reward = 0

        for _ in range(n_steps_per_run):
            action, probs, hidden_activities = network.get_action(state, rng)

            hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
            log_prob = torch.log(probs.squeeze(0)[action])

            new_state, reward, done, _ = env.step(action)
            discounted_reward *= gamma
            discounted_reward += reward
            cum_reward += reward

            el_traces = update_el_traces(el_traces, probs, hidden_activities, action)

            update_weights_online_with_rule(rule=t, network=network, reward=reward, el_traces=el_traces,
                                  log_prob=log_prob, discounted_reward=discounted_reward, done= done,
                                            expected_cum_reward_per_episode=expected_cum_reward_per_episode)

            if done:
                episode_counter += 1
                break
            state = new_state

        # todo: "document" variable
        expected_cum_reward_per_episode = (1-1/n_episodes_reward_expectation)*\
                                          expected_cum_reward_per_episode + \
                                          (1/n_episodes_reward_expectation)*cum_reward/n_steps_per_run
    env.close()
    if mode == 'reward_max':
        return float(cum_reward)
    elif mode == 'episode_min':
        return float(-episode_counter)
    else:
        raise AssertionError('Mode not available')


# Todo: - cache
def objective_one(
    individual: cgp.IndividualSingleGenome,
    n_episodes: int,
    gamma: float,
    seed: int,
    rng: np.random.Generator,
):

    if not individual.fitness_is_None():
        return individual

    # environment initialization
    env = gym.make('CartPole-v0')

    # network initialization
    torch.manual_seed(seed=seed)
    network = Network(n_inputs=env.observation_space.shape[0], n_hidden=100,
                      n_outputs=env.action_space.n, learning_rate=2e-4, weight_update_mode='evolved_rule')

    t = individual.to_torch()
    try:
        individual.fitness = inner_objective(t=t, network=network, env=env,
                                              n_episodes=n_episodes, gamma=gamma,
                                              seed=seed, rng=rng, mode='reward_max')
    except ZeroDivisionError:
        individual.fitness = -np.inf

    # Todo write network.state_dict() to ind (and possibly pickle dumps)
    individual.network = network # assign the trained network to the individual for objective 2
    return individual


def objective_two(
    individual: cgp.IndividualSingleGenome,
    cum_reward_threshold: int,
    gamma: float,
    seed: int,
    rng: np.random.Generator,
):
    if not individual.fitness_is_None():
        return individual

    # environment initialization
    env = gym.make('CartPole-v0')

    t = individual.to_torch()
    try:
        individual.fitness = inner_objective(t=t, network=individual.network, env=env,
                                             cum_reward_threshold=cum_reward_threshold,
                                             gamma=gamma, seed=seed, rng=rng, mode='episode_min')
    except ZeroDivisionError:
        individual.fitness = -np.inf

    return individual


rng = np.random.default_rng(seed=seed)

if use_online_init:
    pop = cgp.Population(**population_params, genome_params=genome_params,
                         individual_init=set_initial_dna)
else:
    pop = cgp.Population(**population_params, genome_params=genome_params)

ea = cgp.ea.MuPlusLambda(**ea_params)

history = {}
history["fitness_champion"] = []
history["expression_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)
    history["expression_champion"].append(pop.champion.to_sympy())


obj_1 = functools.partial(
    objective_one,
    n_episodes=n_episodes,
    gamma=gamma,
    seed=seed,
    rng=rng
)

obj_2 = functools.partial(
    objective_two,
    cum_reward_threshold=cum_reward_threshold,
    gamma=gamma,
    seed=seed,
    rng=rng
)
start = time.time()
cgp.evolve(pop, [obj_1, obj_2], ea, **evolve_params, print_progress=True, callback=recording_callback)
end = time.time()
print(f"Time elapsed:", end - start)

max_fitness = history["fitness_champion"][-1]
best_expr = pop.champion.to_sympy()

print(
    f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})')

