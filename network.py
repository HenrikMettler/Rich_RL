import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from typing import Callable, List, Tuple, Union


class Network(nn.Module):
    """ Network class"""

    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, learning_rate: float,
                 weight_update_mode='autograd')\
            -> None:
        """ Init function
        
        Parameters
        ----------
        n_inputs: int
            Number of network inputs (should be the dimension of the observation space)
        n_hidden: int
            Number of hidden layer neurons
        n_outputs: int
            Number of output neurons (should be the dimension of the action space)
        learning_rate: float
            Learning rate
        """

        super().__init__()
        self.num_actions = n_outputs

        self.hidden_layer = nn.Linear(n_inputs, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_outputs)

        self.learning_rate = learning_rate

        # turn off autograd in output layer, unless using autograd
        if not weight_update_mode == 'autograd':
            for parameter in self.output_layer.parameters():
                parameter.requires_grad = False

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

    def get_action(self, state: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray,
                                                                               torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0)
        hidden_activities, probs = self.forward(Variable(state))
        # Todo: use torch generator instead of numpy rng
        try:
            selected_action = rng.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        except ValueError:
            selected_action = rng.choice(self.num_actions)
        return selected_action, probs, hidden_activities.squeeze()


class VanillaRNN(nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, learning_rate: float,
                 use_autograd=True) \
            -> None:
        """ Init function

        Parameters
        ----------
        n_inputs: int
            Number of network inputs (should be the dimension of the observation space)
        n_hidden: int
            Number of hidden layer neurons
        n_outputs: int
            Number of output neurons (should be the dimension of the action space)
        learning_rate: float
            Learning rate
        use_autograd: bool
            Whether to use autograd parameter updates. Defaults to true
        """

        super().__init__()
        self.num_hidden = n_hidden
        self.num_actions = n_outputs

        self.rnn = nn.RNN(n_inputs, n_hidden)
        self.output_layer = nn.Linear(n_hidden, n_outputs)

        self.learning_rate = learning_rate

        if not use_autograd:
            # turn off autograd for all parameters
            for parameter in self.named_parameters():
                parameter.requires_grad = False
        else:
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
        hidden: torch.Tensor
            Hidden layer activities
        output: torch.Tensor
            Output layer activities
        """

        hidden = self.init_hidden()

        out, hidden = self.rnn(state, hidden)
        output = self.output_layer(out)

        return output, hidden

    def init_hidden(self) -> torch.Tensor:
        return torch.zeros(self.num_hidden)  # Todo: should these be zeros? (is so in most tutorials)
