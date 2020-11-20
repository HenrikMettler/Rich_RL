import gym
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import GAMMA, Network
from typing import Callable, List, Tuple, Union

seed = 1234
torch.manual_seed(seed=seed)
rng = np.random.default_rng(seed=seed)

n_epsisodes: int = 50000
n_steps_max: int = 10000

env = gym.make('CartPole-v0')  # Pendulum-v0
if isinstance(env.action_space, gym.spaces.Box):
    env_is_box = True
else:
    env_is_box = False
env.seed(seed=seed)
env_cont_flag: bool = False
if isinstance(env.action_space, gym.spaces.Box): # must be split since env with discrete action spaces have no attribute shape
    if env.action_space.shape[0] == 1:
        env_cont_flag = True

n_inputs: int = env.observation_space.shape[0]
n_hidden: int = 100
if env_is_box:
    n_outputs: int = env.action_space.shape[0]
else:
    n_outputs: int = env.action_space.n
learning_rate: float = 3e-4

policy_net = Network(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs,
                     learning_rate=learning_rate)


def get_stoch_action_from_output(output_activity: torch.Tensor) -> int:
    softmax = torch.nn.Softmax(dim=0)  # adds a second non-linearity -> too much smoothing?
    output_activity = softmax(output_activity)
    out_numpy = output_activity.detach().numpy()
    return rng.choice(output_activity.shape[0],
                            p=np.squeeze(out_numpy))

n_steps_per_episode: List[int] = []

for episode in range(n_epsisodes):
    observation = env.reset()
    log_probs: List[float] = []
    rewards: List[float] = []

    for steps in range(n_steps_max):
        #env.render()
        _, output_activity = policy_net.forward(observation)
        if env_cont_flag:
            action = output_activity
        else:
            action = rng.choice(output_activity.shape[0], p=np.squeeze(output_activity.detach().numpy())) # get_stoch_action_from_output(output_activity=output_activity)

        # Todo: what is the continuous equivalent of log probs in cont action space?
        #  -> see Lillicrap et al. Cont. Control paper
        log_prob = torch.log(output_activity.squeeze(0)[action])

        observation, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            policy_net.update_with_policy(rewards, log_probs)
            n_steps_per_episode.append(steps)

            if episode % 1000 == 0:
                print(f"episode {episode},  total reward: {np.sum(rewards)},"
                      f" average_reward: {np.mean(rewards)}, length: {steps}\n")

            break
n_steps_smoothed = []
smoothing_coeff = 0.9
n_steps_smoothed.append(n_steps_per_episode[0])
for idx in range(1, len(n_steps_per_episode)):
    n_steps_smoothed.append(smoothing_coeff*n_steps_smoothed[idx-1]+
                            (1-smoothing_coeff)*n_steps_per_episode[idx])
plt.plot(n_steps_per_episode)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.show()