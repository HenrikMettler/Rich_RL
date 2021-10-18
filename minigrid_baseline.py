import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from functions import update_weights, alter_env
from typing import Callable, List, Tuple, Union
from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid
from gym_minigrid.wrappers import FlatObsWrapper, ImgObsWrapper

weight_update_mode = 'autograd'

seed = 123456789
torch.manual_seed(seed=seed)
rng = np.random.default_rng(seed=seed)

n_episodes: int =  100 #50000
n_steps_max: int = 1000
env = DynamicMiniGrid(seed=seed)
env, is_solvable = alter_env(env, n=6)
env = ImgObsWrapper(env)
state = env.respawn()["image"].flatten()

n_inputs: int = np.size(state)
n_hidden: int = 500  # todo: iterate over different sizes
n_outputs: int = 3  # Left, right, forward (pick up, drop, toggle, done are ingnored); env.action_space.n
learning_rate: float = 1e-4  # todo iterate over different lr's

policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                     learning_rate=learning_rate, weight_update_mode=weight_update_mode)

n_steps_per_episode: List[int] = []
rewards_over_episodes: List[float] = []

for episode in range(n_episodes):
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

        if done:
            update_params = {
                "rewards": rewards,
                "probs": probs,
                "log_probs": log_probs,
                "actions": actions,
                "hidden_activities": hidden_activities_all,
            }
            update_weights(network=policy_net, **update_params,
                           weight_update_mode=weight_update_mode)

            n_steps_per_episode.append(steps)

            if episode % 10 == 0:
                print(f"episode {episode}, n_steps: {steps}\n")
                print(f"Episode_reward: {sum(rewards)}")
            break
        state = new_state.flatten()
    rewards_over_episodes.append(sum(rewards))


plt.plot(rewards_over_episodes)
a = 1
