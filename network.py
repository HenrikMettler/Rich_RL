import numpy as np
import torch
import torch.nn as nn

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

    def select_action(self, observation: np.array) -> np.ndarray:
        """ Picks an action from a given observation

        Parameters
        ----------
        observation: np.array
            Current state input from environment

        """
        _, action = self.forward(observation=observation)
        return action.detach().numpy()