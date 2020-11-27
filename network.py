import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tracemalloc

from torch.autograd import Variable
from typing import Callable, List, Tuple, Union

# Reward discount factor
GAMMA: float = 0.9


class Network(nn.Module):
    """ Network class"""

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, learning_rate: float)\
            -> None:
        """ Init function
        
        Parameters
        ----------
        n_inputs: int
            Number of network inputs (should be the dimension of the observation space)
        n_hidden_layer: int
            Number of hidden layer neurons
        n_outputs: 
            Number of output neurons (should be the dimension of the action space)
            :param learning_rate:
        """

        super(Network, self).__init__()
        self.num_actions = n_outputs

        self.hidden_layer = nn.Linear(n_inputs, n_hidden) # Todo: train biases as well?
        self.output_layer = nn.Linear(n_hidden, n_outputs)

        self.learning_rate = learning_rate

        # optimizer for baseline with policy gradient
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute a forward pass in the network and return output and hidden activities

        Parameters
        ----------
        state: torch.Tensor
            Current state input from environment

        Returns
        -------
        hidden_activities: torch.Tensor
            Hidden layer activities
        output_activities: torch.Tensor
            Output layer activities
        """

        # relu for hidden and softmax for output as in:
        # https://medium.com/@thechrisyoon/
        # deriving-policy-gradients-and-implementing-reinforce-f887949bd63
        hidden_activities: torch.Tensor = F.relu(self.hidden_layer(state))
        output_activities: torch.Tensor = F.softmax(
            self.output_layer(hidden_activities), dim=1)

        return hidden_activities, output_activities

    def get_action(self, state: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0)
        _, probs = self.forward(Variable(state))
        selected_action = rng.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[selected_action])
        return selected_action, log_prob


def update_weights(
    network: Network,
    t: torch.nn.Module,
    observation: np.ndarray,
    hidden_activities: torch.Tensor,
    output_activities: torch.Tensor,
    reward: float,
) -> None:
    """

    :param t: torch.nn.Module, update func as torch module
    :param observation: numpy.ndarray, input (from env)
    :param hidden_activities: torch.tensor,  activity of hidden-layer neurons
    :param output_activities: torch.tensor activity of output-layer neurons
    :param reward: float, reward in the trial
    :param learning_rate: float, multiplication factor for update
    :return: None
    """
    # implementation with torch

    # ugly stuff to have data in correct shape
    n_repeats = hidden_activities.shape[-1]
    reward_repeated = _repeat_unsqueeze_tensor(network=network,
        element=reward, n_repeats=n_repeats
    )
    hidden_activities = hidden_activities.unsqueeze(dim=1)

    # first layer update
    for idx_obs, observation_element in enumerate(observation):

        # ugly stuff to make the inputs right shape and dtype
        observation_repeated = _repeat_unsqueeze_tensor(network=network,
            element=observation_element, n_repeats=n_repeats
        )

        weights = network.hidden_layer.weight[:, idx_obs]
        weights = weights.unsqueeze(dim=1)
        update = _calculate_update_weight_tensor(
            network=network,
            t=t,
            pre=observation_repeated,
            post=hidden_activities,
            weights=weights,
            rewards=reward_repeated,
        )
        network.hidden_layer.weight[:, idx_obs] += update

    # second layer update
    output_activities_repeated = output_activities.repeat(n_repeats)
    output_activities_repeated = output_activities_repeated.unsqueeze(dim=1)
    weights = network.output_layer.weight.T
    update = _calculate_update_weight_tensor(
        network=network,
        t=t,
        pre=hidden_activities,
        post=output_activities_repeated,
        weights=weights,
        rewards=reward_repeated
    )
    network.output_layer.weight[:] += update.T

    """# element-wise implementation
    # first layer update
    for idx_obs, observation_element in enumerate(observation):
        for hidden_activity, weight in zip(hidden_activities,
                                           network.hidden_layer.weight[:, idx_obs]):
            weight += learning_rate * f([observation_element, hidden_activity,
                                         weight, reward])[0]
    # second layer update Todo: only works with singe output!
    for hidden_activity, output_weight in zip(hidden_activities, network.output_layer.weight):
        output_weight += learning_rate*f([hidden_activity, output_activities,
                                          output_weight, reward])[0]
    """


def _calculate_update_weight_tensor(
    network,
    t: torch.nn.Module,
    pre: torch.Tensor,
    post: torch.Tensor,
    weights: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:

    input_variables = torch.cat([pre, post, weights, rewards], dim=1)
    output = t(input_variables)
    update = network.learning_rate * output
    return update.squeeze(dim=1)


def _repeat_unsqueeze_tensor(
    network, element: Union[float, int], n_repeats: int, datatype=torch.float32
) -> torch.Tensor:
    """ repeat scalar elements and unsqueeze them

    Parameters
    ----------
    element: Union[float, int]
        element to be repeated
    n_repeats: int
        number of times the element should be repeated
    datatype:
        desired datatype of the output

    Return
    ------
    torch.Tensor of size n_repeats x 1, with all values equal to element
        and converted to datatype
    """
    element_repeated = torch.tensor(element, dtype=datatype).repeat(n_repeats)
    element_repeated_unsqueezed = element_repeated.unsqueeze(dim=1)
    return element_repeated_unsqueezed


def calculate_discounted_reward(rewards: List[float]):
    discounted_rewards_list: List[float] = []

    for t in range(len(rewards)):
        discounted_reward: float = 0
        exponent: int = 0
        for reward in rewards[t:]:
            discounted_reward += GAMMA**exponent * reward
            exponent += 1
        discounted_rewards_list.append(discounted_reward)

    discounted_rewards: torch.Tensor = torch.Tensor(discounted_rewards_list)
    return discounted_rewards


def normalize_discounted_rewards(discounted_rewards: torch.Tensor):
    return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)


def calculate_policy_gradient_element_wise(log_probs: List[torch.Tensor], discounted_rewards: torch.Tensor):
    policy_gradient_list: List[torch.Tensor] = []
    for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
        # -, since we do gradient ascent on the expected discounted rewards, not descent
        policy_gradient_list.append(-log_prob*discounted_reward)
    return policy_gradient_list


def update_with_policy(network: Network, rewards: List[float], log_probs: List[torch.Tensor]):
    """adapted from: https://medium.com/@thechrisyoon/
    deriving-policy-gradients-and-implementing-reinforce-f887949bd63"""

    discounted_rewards = calculate_discounted_reward(rewards)
    # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
    discounted_rewards = normalize_discounted_rewards(discounted_rewards)

    policy_gradient_list = calculate_policy_gradient_element_wise(log_probs=log_probs,
                                                                  discounted_rewards=discounted_rewards)

    network.optimizer.zero_grad()
    policy_gradient: torch.Tensor = torch.stack(policy_gradient_list).sum()
    policy_gradient.backward()
    network.optimizer.step()


def update_weight_with_policy_gradient(network: Network, rewards: List[float], log_probs: List[torch.Tensor]):
    """ mimic unsupervised learning with gradient"""

    discounted_rewards = calculate_discounted_reward(rewards)
    # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
    discounted_rewards = normalize_discounted_rewards(discounted_rewards)
    policy_gradient_list = calculate_policy_gradient_element_wise(log_probs=log_probs,
                                                                  discounted_rewards=discounted_rewards)