import numpy as np
import torch
import torch.nn as nn
import tracemalloc

from typing import Callable, Tuple, Union


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
        """

        :param f: Callable, update func for element-wise update
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
                learning_rate=learning_rate,
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
            learning_rate=learning_rate,
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
        learning_rate: float,
    ) -> torch.Tensor:

        input_variables = torch.cat([pre, post, weights, rewards], dim=1)
        output = t(input_variables)
        update = learning_rate * output
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
