import torch
import numpy as np
import pytest
import sys
import ipdb

sys.path.insert(0, '../')
from network import Network
from functions import update_weights, update_weights_online, \
    compute_weight_bias_updates_equation2, compute_weight_bias_updates_equation4, \
    calculate_discounted_rewards, calc_el_traces, normalize_discounted_rewards, update_el_traces


def test_comparison_torch_eq2(seed, network_params, states, rewards):

    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    net = Network(**network_params, weight_update_mode='autograd')

    actions = []
    log_probs = []
    probs = []
    hidden_activities_all = []

    for step in range(2):
        action, prob, hidden_activities = net.get_action(states[step], rng)

        hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
        log_prob = torch.log(prob.squeeze(0)[action])

        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        hidden_activities_all.append(hidden_activities)

    discounted_rewards = normalize_discounted_rewards(calculate_discounted_rewards(rewards))
    weight_updates = torch.empty_like(net.output_layer.weight)
    bias_updates = torch.empty_like(net.output_layer.bias)
    with torch.no_grad():
        for idx_action in range(network_params["n_outputs"]):
            updates = compute_weight_bias_updates_equation2(actions, idx_action,
                                                  discounted_rewards, probs, hidden_activities_all)
            weight_updates[idx_action] = updates[0][:-1]
            bias_updates[idx_action] = updates[0][-1]

    update_weights(network=net, rewards=rewards, log_probs=log_probs,
                       actions=actions,probs=probs, hidden_activities=hidden_activities_all)

    assert weight_updates.numpy() != pytest.approx(0.)
    assert bias_updates.numpy() != pytest.approx(0.)
    assert net.output_layer.weight._grad.detach().numpy() == pytest.approx(weight_updates.numpy())
    assert net.output_layer.bias._grad.detach().numpy() == pytest.approx(bias_updates.numpy())


def test_comparison_torch_eq4(seed, network_params, states, rewards):

    torch.manual_seed(seed=seed)
    rng = np.random.default_rng(seed=seed)

    net = Network(**network_params, weight_update_mode='autograd')

    actions = []
    log_probs = []
    probs = []
    hidden_activities_all = []

    for step in range(2):
        action, prob, hidden_activities = net.get_action(states[step], rng)

        hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
        log_prob = torch.log(prob.squeeze(0)[action])

        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        hidden_activities_all.append(hidden_activities)

    el_traces = calc_el_traces(probs=probs, hidden_activities=hidden_activities_all, actions=actions)

    weight_updates = torch.empty_like(net.output_layer.weight)
    bias_updates = torch.empty_like(net.output_layer.bias)
    with torch.no_grad():
        for idx_action in range(network_params["n_outputs"]):
            weight_updates[idx_action], bias_updates[idx_action] = compute_weight_bias_updates_equation4(rewards, el_traces[idx_action])

    update_weights(network=net, rewards=rewards, log_probs=log_probs, actions=actions, probs=probs,
                   hidden_activities=hidden_activities_all, normalize_discounted_rewards_b=False)

    assert weight_updates.numpy() != pytest.approx(0.)
    assert bias_updates.numpy() != pytest.approx(0.)
    #print(net.output_layer.weight._grad.detach().numpy()/weight_updates.numpy())
    assert net.output_layer.weight._grad.detach().numpy() == pytest.approx(weight_updates.numpy())
    assert net.output_layer.bias._grad.detach().numpy() == pytest.approx(bias_updates.numpy())


# Todo: test online update
# def test_update_weights_online(seed, network_params, states, rewards, gamma):
#
#     torch.manual_seed(seed=seed)
#     rng = np.random.default_rng(seed=seed)
#
#     for step in range(1):
#         net = Network(**network_params, weight_update_mode='autograd')
#         online_net = Network(**network_params, weight_update_mode='autograd')
#
#         actions = []
#         log_probs = []
#         probs = []
#         hidden_activities_all = []
#         el_traces = torch.zeros([network_params["n_outputs"],network_params["n_hidden"] + 1])
#         discounted_reward = 0
#
#         action, prob, hidden_activities = net.get_action(states[step], rng)
#         hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
#         log_prob = torch.log(prob.squeeze(0)[action])
#
#         discounted_reward *= gamma
#         discounted_reward += rewards[step]
#
#         log_probs.append(log_prob)
#         probs.append(prob)
#         actions.append(action)
#         hidden_activities_all.append(hidden_activities)
#
#         update_params = {
#             "rewards": rewards,
#             "probs": probs,
#             "log_probs": log_probs,
#             "actions": actions,
#             "hidden_activities": hidden_activities_all,
#         }
#
#         update_weights(network=net, **update_params, normalize_discounted_rewards_b=False)
#
#         el_traces = update_el_traces(el_traces, prob, hidden_activities, action)
#
#         update_weights_online(network=online_net, reward=rewards[step],  el_traces=el_traces,
#                               log_prob=log_prob, discounted_reward = discounted_reward)
#
#         assert net.output_layer.weight.numpy() == pytest.approx(online_net.output_layer.weight.numpy())

