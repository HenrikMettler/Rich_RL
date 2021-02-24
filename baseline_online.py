import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import GAMMA, Network, update_weights_online
from typing import Callable, List, Tuple, Union

seed = 123
torch.manual_seed(seed=seed)
rng = np.random.default_rng(seed=seed)

n_epsisodes: int = 5000
n_steps_max: int = 10000

env = gym.make('CartPole-v0')  # Pendulum-v0
if isinstance(env.action_space, gym.spaces.Box):
    env_is_box = True
else:
    env_is_box = False
env.seed(seed=seed)
env_cont_flag: bool = False


n_inputs: int = env.observation_space.shape[0]
n_hidden: int = 100
if env_is_box:
    n_outputs: int = env.action_space.shape[0]
else:
    n_outputs: int = env.action_space.n
learning_rate: float = 2e-4 #3e-4

online_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                     learning_rate=learning_rate)


n_steps_per_episode: List[int] = []

for episode in range(n_epsisodes):
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []

    for steps in range(n_steps_max):
        action, log_prob = online_net.get_action(state, rng)

        new_state, reward, done, _ = env.step(action)  # todo replace done with
        log_probs.append(log_prob)
        rewards.append(reward)

        update_weights_online(network=online_net, rewards=rewards, log_probs=log_probs)
        state = new_state

n_steps_smoothed = []
smoothing_coeff = 0.9
n_steps_smoothed.append(n_steps_per_episode[0])
for idx in range(1, len(n_steps_per_episode)):
    n_steps_smoothed.append(smoothing_coeff*n_steps_smoothed[idx-1]+
                            (1-smoothing_coeff)*n_steps_per_episode[idx])
plt.plot(n_steps_per_episode, label='Duration')
plt.plot(n_steps_smoothed, label='Smoothed Duration')
plt.xlabel('Episode')
plt.ylabel('Duration')

plt.show()