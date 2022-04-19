import numpy as np
import gym
import sympy
import cgp

from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid


def plot_environments(seed, max_n_alterations, n_alterations_per_new_env, prob_alteration_dict):
    env = DynamicMiniGrid(seed=seed)
    for n_alter in range(1, max_n_alterations):
        env.alter_env(env=env, n=n_alterations_per_new_env, prob_alteration_dict=prob_alteration_dict)
        env.render()