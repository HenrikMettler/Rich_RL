import numpy as np
import gym
import torch
import sympy
import functools
import warnings
import time
import cProfile

import cgp

from network import Network
from functions import update_el_traces, update_weights_online_with_rule
from variables import onlinelr_dna_20internal
from operators import Const05Node, Const2Node

gamma = 0.9
cumulative_reward_threshold = 250000  # empirical value that policy gradient receives after ~ 3000 episodes

use_online_init = True


def set_initial_dna(ind):
    genome = cgp.Genome(**genome_params)
    genome.dna = onlinelr_dna_20internal
    return cgp.IndividualSingleGenome(genome)


def inner_objective(
    t: torch.nn.Module,
    network: Network,
    env: gym.Env,
    n_steps_per_run: int,
    seed: int,
    rng: np.random.Generator,
) -> float:

    env.seed(seed)
    cum_reward = 0
    episode_counter = 0
    while cum_reward < cumulative_reward_threshold:
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
                                  log_prob=log_prob, discounted_reward=discounted_reward, done= done)

            if done:
                episode_counter += 1
                break
            state = new_state

    env.close()

    return float(episode_counter)


def objective(
    individual: cgp.IndividualSingleGenome,
    n_steps_per_run: int,
    learning_rate: float,
    seed: int,
    rng: np.random.Generator,
):

    if not individual.fitness_is_None():
        return individual

    # environment initialization
    env = gym.make('CartPole-v0')

    # network initialization
    torch.manual_seed(seed=seed)
    n_inputs = env.observation_space.shape[0]
    n_hidden = 100
    n_outputs: int = env.action_space.n

    network = Network(n_inputs, n_hidden, n_outputs, learning_rate,
                 weight_update_mode='evolved_rule')

    t = individual.to_torch()
    try:
        individual.fitness = -inner_objective(t=t, network=network, env=env,
                                                 n_steps_per_run=n_steps_per_run, seed=seed,
                                                 rng=rng)
    except ZeroDivisionError:
        individual.fitness = -np.inf

    return individual


seed = 123
n_steps_per_run = 200
learning_rate = 2e-4
rng = np.random.default_rng(seed=seed)

# population and evolutionary algorithm initialization
population_params = {"n_parents": 1, "seed": seed}
genome_params = {
    "n_inputs": 3,  # reward, el_traces, done (episode termination)
    "n_outputs": 1,
    "n_columns": 20,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (cgp.Mul, cgp.Add, cgp.Sub, cgp.ConstantFloat, Const05Node, Const2Node) #(cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}
ea_params = {"n_offsprings": 4, "mutation_rate": 0.03, "reorder_genome": True, "n_processes": 1}
evolve_params = {"max_generations": 100, "termination_fitness": -2000}  # Todo: set reasonable termination fitness

if use_online_init:
    pop = cgp.Population(**population_params, genome_params=genome_params, individual_init=set_initial_dna)
else:
    pop = cgp.Population(**population_params, genome_params=genome_params)

ea = cgp.ea.MuPlusLambda(**ea_params)

# initialize a history
history = {}
history["fitness_champion"] = []
# history["expression_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)
    # history["expression_champion"].append(pop.champion.to_sympy())


obj = functools.partial(
    objective,
    n_steps_per_run=n_steps_per_run,
    learning_rate=learning_rate,
    seed=seed,
    rng=rng
)

start = time.time()
cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)
end = time.time()
print(f"Time elapsed:", end - start)

max_fitness = history["fitness_champion"][-1]
best_expr = pop.champion.to_sympy()
# best_expr = best_expr.replace("x_0", "pre").replace("x_1", "post").replace("x_2, "weight")
# .replace("x_4", "reward")
print(
    f'Learning rule with highest fitness: "{best_expr}" (fitness: {max_fitness})')
