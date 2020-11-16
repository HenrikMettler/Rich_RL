import numpy as np
import torch
import torch.nn as nn
import tracemalloc

from typing import Callable, Tuple


class Network(nn.Module):
    """ Network class"""

    def __init__(self, n_inputs: int, n_hidden_layer: int, n_outputs: int) -> None:
        """ Init function
        
        Parameters
        ----------
        n_inputs: int
            Number of network inputs (should be the dimension of the observation space)
        n_hidden_layer: int
            Number of hidden layer neurons
        n_outputs: 
            Number of output neurons (should be the dimension of the action space)
        """

        super(Network, self).__init__()

        self.hidden_layer = nn.Linear(n_inputs, n_hidden_layer, bias=False)
        self.output_layer = nn.Linear(n_hidden_layer, n_outputs, bias=False)

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

        # linear activation function
        hidden_activities: torch.Tensor = self.hidden_layer(state)
        output_activities: torch.Tensor = self.output_layer(hidden_activities)

        return hidden_activities, output_activities

    def update_weights(
        self,
        f: Callable,
        t: torch.nn.Module,
        observation: np.ndarray,
        hidden_activities: torch.Tensor,
        output_activities: torch.Tensor,
        reward: float,
        learning_rate: float,
    ) -> None:

        # implementation with torch

        # ugly stuff to have data in correct shape
        reward_repeated = torch.tensor(reward).repeat(hidden_activities.shape[0]).unsqueeze(dim=1)
        hidden_activities = hidden_activities.unsqueeze(dim=1)

        # first layer update
        for idx_obs, observation_element in enumerate(observation):

            # ugly stuff to make the inputs right shape and dtype
            observation_repeated = torch.tensor(observation_element, dtype=torch.float32).\
                 repeat(hidden_activities.shape[0])
            observation_repeated = observation_repeated.unsqueeze(dim=1)
            weights = self.hidden_layer.weight[:, idx_obs]
            weights = weights.unsqueeze(dim=1)
            input_variables = torch.cat([observation_repeated, hidden_activities,
                                          weights, reward_repeated], dim=1)

            output = t(input_variables).squeeze(dim=1)
            update = learning_rate * output
            self.hidden_layer.weight[:, idx_obs] += update

        # second layer update
        output_activities_repeated = output_activities.repeat(hidden_activities.shape[0])
        output_activities_repeated = output_activities_repeated.unsqueeze(dim=1)
        weights = self.output_layer.weight.T
        input_variables = torch.cat([hidden_activities, output_activities_repeated, weights, reward_repeated],
                           dim=1)
        output = t(input_variables)
        update = learning_rate * output
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
