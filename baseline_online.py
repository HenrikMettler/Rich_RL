import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from functions import update_el_traces, update_weights_online
from typing import Callable, List, Tuple, Union

gamma = 0.9

# Todo: Write params in json file
seed = 123
torch.manual_seed(seed=seed)
rng = np.random.default_rng(seed=seed)

n_epsisodes: int = 10000
n_steps_max: int = 200

env = gym.make('CartPole-v0')  # Pendulum-v0
env.seed(seed=seed)

n_inputs: int = env.observation_space.shape[0]
n_hidden: int = 100
n_outputs: int = env.action_space.n
learning_rate: float = 2e-4

online_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                     learning_rate=learning_rate)


n_steps_per_episode: List[int] = []

for episode in range(n_epsisodes):
    state = env.reset()
    el_traces = torch.zeros([n_outputs, n_hidden+1])
    discounted_reward = 0

    for steps in range(n_steps_max):

        action, probs, hidden_activities = online_net.get_action(state, rng)

        hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
        log_prob = torch.log(probs.squeeze(0)[action])

        new_state, reward, done, _ = env.step(action)
        # Reward modulation
        if not done:
            reward = 0.0 # reward  = 0
        else:
            reward = -1.0
        discounted_reward *= gamma
        discounted_reward += reward

        el_traces = update_el_traces(el_traces, probs, hidden_activities, action)

        update_weights_online_with_policy_gradient(network=online_net, reward=reward,  el_traces=el_traces,
                              log_prob=log_prob, discounted_reward = discounted_reward)

        if done:
            n_steps_per_episode.append(steps)
            if episode % 100 == 0:
                print(f"episode {episode}, n_steps: {steps}\n")
            break
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
plt.title(f"Online weight update")

plt.show()
a=1