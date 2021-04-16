import torch
import numpy as np
import pytest
import sys

sys.path.insert(0, '../')
from network import Network, update_with_policy, compute_weight_bias_updates_equation2, \
    calculate_discounted_rewards, normalize_discounted_rewards


# Todo: test online update
def test_comparison_torch_eq2(seed, network_params, states, rewards):

    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    net = Network(**network_params, use_autograd_for_output=True)

    actions = []
    log_probs = []
    probs = []
    hidden_activities_all = []

    for step in range(2):
        action, log_prob, prob, hidden_activities = net.get_action(states[step], rng)
        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        hidden_activities_all.append(hidden_activities)

    discounted_rewards = normalize_discounted_rewards(calculate_discounted_rewards(rewards))
    weight_updates = torch.empty_like(net.output_layer.weight)
    bias_updates = torch.empty_like(net.output_layer.bias)
    with torch.no_grad():
        for idx_action in range(network_params["n_outputs"]):
            weight_updates[idx_action], bias_updates[idx_action] = compute_weight_bias_updates_equation2(actions, idx_action,
                                                  discounted_rewards, probs, hidden_activities_all)

    update_with_policy(network=net, rewards=rewards, log_probs=log_probs,
                       use_autograd_for_output=True, actions=actions,
                       probs=probs, hidden_activities=probs)

    assert weight_updates.numpy() != pytest.approx(0.)
    assert bias_updates.numpy() != pytest.approx(0.)
    assert net.output_layer.weight._grad.detach().numpy() == pytest.approx(weight_updates.numpy())
    assert net.output_layer.bias._grad.detach().numpy() == pytest.approx(bias_updates.numpy())


def test_comparison_torch_eq4(seed, network_params, states, rewards):

    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    net = Network(**network_params, use_autograd_for_output=True)
    actions = []
    log_probs = []
    probs = []
    hidden_activities_all = []

    for step in range(2):
        action, log_prob, prob, hidden_activities = net.get_action(states[step], rng)
        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        hidden_activities_all.append(hidden_activities)

    discounted_rewards = normalize_discounted_rewards(calculate_discounted_rewards(rewards))
    weight_updates = torch.empty_like(net.output_layer.weight)
    bias_updates = torch.empty_like(net.output_layer.bias)
    with torch.no_grad():
        raise NotImplementedError

    update_with_policy(network=net, rewards=rewards, log_probs=log_probs,
                       use_autograd_for_output=True, actions=actions,
                       probs=probs, hidden_activities=probs)

    assert weight_updates.numpy() != pytest.approx(0.)
    assert bias_updates.numpy() != pytest.approx(0.)
    assert net.output_layer.weight._grad.detach().numpy() == pytest.approx(weight_updates.numpy())
    assert net.output_layer.bias._grad.detach().numpy() == pytest.approx(bias_updates.numpy())

