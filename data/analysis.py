import numpy as np
import gym
import sympy
import cgp
import os
import pickle
import matplotlib.pyplot as plt
from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid
from helperfunctions import plot_environments

data_directory= 'n_hidden_scan_with_no_backprop'
data_foldername = 'b91bc81d9e0d1e45ff7aa5200ab586e2'

fitness_array = []
n_hidden_array = []
subfoldernames = os.listdir(data_directory)
for subfoldername in subfoldernames:
    if subfoldername != '.DS_Store':
        with open(f"{data_directory}/{subfoldername}/params.pickle", 'rb') as f:
            params = pickle.load(f)
        with open(f"{data_directory}/{subfoldername}/history.pickle", 'rb') as f:
            history = pickle.load(f)
        fitness_array.append(history['fitness_champion'][-1])
        n_hidden_array.append(params['network_params']['n_hidden'])

plt.scatter(n_hidden_array, fitness_array)
plt.xlabel('n hidden')
plt.ylabel('fitness')
plt.title('frozen input weights, output weights with PG (4 curr, 3 env)')

network_params = params['network_params']
env_params = params['env_params']
prob_alteration_dict = params['prob_alteration_dict']
seeds = env_params["seeds"]
max_n_alterations = env_params["max_n_alterations"]
n_alterations_per_new_env = env_params["n_alterations_per_new_env"]
for seed in seeds:
    plot_environments(int(seed), max_n_alterations, n_alterations_per_new_env, prob_alteration_dict)


