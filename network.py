import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tracemalloc

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

        self.hidden_layer = nn.Linear(n_inputs, n_hidden, bias=False) # Todo: train biases as well?
        self.output_layer = nn.Linear(n_hidden, n_outputs, bias=False)

        self.learning_rate = learning_rate

        # optimizer for baseline with policy gradient
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, observation: np.array) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute a forward pass in the network and return output and hidden activities

        Parameters
        ----------
        observation: np.array
            Current state input from environment

        Returns
        -------
        hidden_activities: torch.Tensor
            Hidden layer activities
        output_activities: torch.Tensor
            Output layer activities
        """

        state = torch.as_tensor(observation).float().detach()

        # tanh activation function as in Najarro & Risi
        # (https://proceedings.neurips.cc/paper/2020/hash/
        # ee23e7ad9b473ad072d57aaa9b2a5222-Abstract.html)
        hidden_activities: torch.Tensor = torch.relu(self.hidden_layer(state))
        output_activities: torch.Tensor = torch.softmax(
            self.output_layer(hidden_activities), dim=0)

        return hidden_activities, output_activities

    def update_weights(
        self,
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
        n_repeats = hidden_activities.shape[0]
        reward_repeated = self._repeat_unsqueeze_tensor(
            element=reward, n_repeats=n_repeats
        )
        hidden_activities = hidden_activities.unsqueeze(dim=1)

        # first layer update
        for idx_obs, observation_element in enumerate(observation):

            # ugly stuff to make the inputs right shape and dtype
            observation_repeated = self._repeat_unsqueeze_tensor(
                element=observation_element, n_repeats=n_repeats
            )

            weights = self.hidden_layer.weight[:, idx_obs]
            weights = weights.unsqueeze(dim=1)
            update = self._calculate_update_weight_tensor(
                t=t,
                pre=observation_repeated,
                post=hidden_activities,
                weights=weights,
                rewards=reward_repeated,
            )
            self.hidden_layer.weight[:, idx_obs] += update

        # second layer update
        output_activities_repeated = output_activities.repeat(n_repeats)
        output_activities_repeated = output_activities_repeated.unsqueeze(dim=1)
        weights = self.output_layer.weight.T
        update = self._calculate_update_weight_tensor(
            t=t,
            pre=hidden_activities,
            post=output_activities_repeated,
            weights=weights,
            rewards=reward_repeated,
        )
        self.output_layer.weight[:] += update.T

        """# element-wise implementation
        # first layer update
        for idx_obs, observation_element in enumerate(observation):
            for hidden_activity, weight in zip(hidden_activities,
                                               self.hidden_layer.weight[:, idx_obs]):
                weight += learning_rate * f([observation_element, hidden_activity,
                                             weight, reward])[0]
        # second layer update Todo: only works with singe output!
        for hidden_activity, output_weight in zip(hidden_activities, self.output_layer.weight):
            output_weight += learning_rate*f([hidden_activity, output_activities,
                                              output_weight, reward])[0]
        """

    def _calculate_update_weight_tensor(
        self,
        t: torch.nn.Module,
        pre: torch.Tensor,
        post: torch.Tensor,
        weights: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:

        input_variables = torch.cat([pre, post, weights, rewards], dim=1)
        output = t(input_variables)
        update = self.learning_rate * output
        return update.squeeze(dim=1)

    def _repeat_unsqueeze_tensor(
        self, element: Union[float, int], n_repeats: int, datatype=torch.float32
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

    def update_with_policy(self, rewards: List[int], log_probs: List[int]):
        """adapted from: https://medium.com/@thechrisyoon/
        deriving-policy-gradients-and-implementing-reinforce-f887949bd63"""
        discounted_rewards_list: List[float] = []

        for t in range(len(rewards)):
            discounted_reward: float = 0
            exponent: int = 0
            for reward in rewards:
                discounted_reward += GAMMA**exponent * reward
                exponent += 1
            discounted_rewards_list.append(discounted_reward)

        discounted_rewards: torch.Tensor = torch.Tensor(discounted_rewards_list)
        # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / \
                             (discounted_rewards.std() + 1e-9)


        policy_gradient_list: List[float] = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            # -, since we do gradient ascent on the expected discounted rewards, not descent
            policy_gradient_list.append(log_prob*discounted_reward)

        policy_gradient: torch.Tensor = torch.stack(policy_gradient_list).sum()
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()


