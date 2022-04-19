import torch
import numpy as np
import pytest
import sys

from network import Network, VanillaRNN


def test_network_init(network_params):

    net = Network(**network_params)

    assert net.learning_rate_hid2out == network_params['learning_rate_hid2out']
    assert net.learning_rate_inp2hid == network_params['learning_rate_inp2hid']
    assert net.hidden_layer.in_features == network_params['n_inputs']
    assert net.hidden_layer.out_features == network_params['n_hidden']
    assert net.output_layer.out_features == network_params['n_outputs']
    assert net.beta == network_params['beta']


def test_forward(network_params, states):

    net = Network(**network_params)
    state1 = states[0]
    state2 = states[1]

    hidden_activities = np.dot(net.hidden_layer.weight.detach().numpy(), state1) + \
                        net.hidden_layer.bias.detach().numpy()
    hidden_activities = [i if i > 0 else 0.0 for i in hidden_activities]

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))

    output_activities = np.dot(net.output_layer.weight.detach().numpy(), hidden_activities)+net.output_layer.bias.detach().numpy()
    output_activities = softmax(output_activities)

    hidden_activities_net, output_activities_net = net.forward(state=torch.from_numpy(state1).float().unsqueeze(0))

    assert torch.tensor(hidden_activities) == pytest.approx(hidden_activities_net)
    assert torch.tensor(output_activities) == pytest.approx(output_activities_net)

    hidden_activities = np.dot(net.hidden_layer.weight.detach().numpy(), state2) + \
                        net.hidden_layer.bias.detach().numpy()
    hidden_activities = [i if i > 0 else 0.0 for i in hidden_activities]

    output_activities = np.dot(net.output_layer.weight.detach().numpy(), hidden_activities)+net.output_layer.bias.detach().numpy()
    output_activities = softmax(output_activities)

    hidden_activities_net, output_activities_net = net.forward(state=torch.from_numpy(states[0]).float().unsqueeze(0))

    assert hidden_activities == pytest.approx(hidden_activities_net.detach().numpy()[0])
    assert output_activities == pytest.approx(output_activities_net.detach().numpy()[0])