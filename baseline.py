import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from functions import update_weights
from typing import Callable, List, Tuple, Union

weight_update_mode = 'equation4'  # options: 'autograd', 'equation2', 'equation4' # todo: rename modes; add online

seed = 123
torch.manual_seed(seed=seed)
rng = np.random.default_rng(seed=seed)

n_epsisodes: int = 5000
n_steps_max: int = 10000

env = gym.make('CartPole-v0')
env.seed(seed=seed)

n_inputs: int = env.observation_space.shape[0]
n_hidden: int = 100
n_outputs: int = env.action_space.n
learning_rate: float = 2e-4 #3e-4

policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                     learning_rate=learning_rate, weight_update_mode=weight_update_mode)

n_steps_per_episode: List[int] = []

for episode in range(n_epsisodes):
    state = env.reset()
    log_probs: List[torch.Tensor] = []
    probs: List[float] = []
    actions: List[int] = []
    rewards: List[float] = []
    hidden_activities_all = []

    for steps in range(n_steps_max):

        action, prob, hidden_activities = policy_net.get_action(state, rng)

        # Todo: Discuss with Jay, if catting 1 here is a good solution
        hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
        log_prob = torch.log(prob.squeeze(0)[action])

        new_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        rewards.append(reward)
        hidden_activities_all.append(hidden_activities)

        if done:
            update_params={
                "rewards": rewards,
                "probs": probs,
                "log_probs": log_probs,
                "actions": actions,
                "hidden_activities": hidden_activities_all,
            }
            update_weights(network=policy_net, **update_params,
                           weight_update_mode=weight_update_mode)

            n_steps_per_episode.append(steps)

            if episode % 100 == 0:
                print(f"episode {episode}, n_steps: {steps}\n")

            break
        state = new_state

# plotting episode duration
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
plt.title(f"Output layer update mode: {weight_update_mode}")

plt.show()
