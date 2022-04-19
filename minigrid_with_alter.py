import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from network import Network
from functions import update_weights, alter_env
from typing import Callable, List, Tuple, Union
from gym_minigrid.envs.dynamic_minigrid import DynamicMiniGrid
from gym_minigrid.wrappers import ImgObsWrapper

if __name__ == "__main__":

    seeds = np.linspace(1234567890, 1234567899, 9)

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)
    weight_update_mode = params['weight_update_mode']

    max_n_alterations = 10
    network_reset_after_alteration = params['network_reset_after_alteration']
    prob_alteration_dict = params['prob_alteration_dict']
    n_episodes_per_alteration: int = 2000
    n_steps_max: int = 100

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
        learning_rate: float = 0.01

        policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                             learning_rate=learning_rate, weight_update_mode=weight_update_mode)

        n_steps_per_episode: List[int] = []
        rewards_over_episodes: List[float] = []

        for n_alter in range(1, max_n_alterations):

            # environement altering
            env = alter_env(env=env, n=1, prob_alteration_dict=prob_alteration_dict)
            env.render()
            env_name = f'env_seed{seed}_nAlterations{n_alter}'
            plt.savefig(env_name)

            if network_reset_after_alteration:
                # Todo: check whether this always uses the same seed
                policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                             learning_rate=learning_rate, weight_update_mode=weight_update_mode)

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
                        update_weights(network=policy_net, weight_update_mode=weight_update_mode,
                                       normalize_discounted_rewards_b=False, **update_params)

                        n_steps_per_episode.append(steps)
                        break

                    state = new_state.flatten()
                rewards_over_episodes.append(sum(rewards))

            env.respawn()

        save_data = {
            'rewards_over_episodes': rewards_over_episodes,
            'n_steps_per_episode': n_steps_per_episode,
            'network': policy_net,
        }

        data_name = f'data_env_reset{network_reset_after_alteration}_seed{seed}.pickle'
        with open(data_name, 'wb') as f:
            pickle.dump(save_data, f)