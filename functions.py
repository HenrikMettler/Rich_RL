import torch
import numpy as np

from network import Network
from typing import AnyStr, Callable, Dict, List, Tuple, Union


def calculate_discounted_rewards(rewards: List[float], gamma=0.9):
    rewards = torch.Tensor(rewards)
    N = len(rewards)
    discounted_rewards = torch.empty(N)

    for t in range(N):
        gamma_factor = gamma**torch.arange(N-t, dtype=torch.float)
        discounted_rewards[t] = gamma_factor @ rewards[t:]

    return discounted_rewards


def normalize_discounted_rewards(discounted_rewards: torch.Tensor):
    if len(discounted_rewards) == 0:
        raise RuntimeError("Length of discounted rewards must be non-zero")
    elif len(discounted_rewards) == 1:
        return torch.Tensor([0.])
    return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-12)


def calculate_policy_gradient_element_wise(log_probs: List[torch.Tensor], discounted_rewards: torch.Tensor):
    policy_gradient_list: List[torch.Tensor] = []
    for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
        # -, since we do gradient ascent on the expected discounted rewards, not descent
        policy_gradient_list.append(-log_prob*discounted_reward)
    return policy_gradient_list


def update_weights(network: Network, rewards, log_probs, probs, actions, hidden_activities,
                           weight_update_mode: AnyStr):
    """adapted from: https://medium.com/@thechrisyoon/
    deriving-policy-gradients-and-implementing-reinforce-f887949bd63"""

    discounted_rewards = calculate_discounted_rewards(rewards)
    # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
    discounted_rewards = normalize_discounted_rewards(discounted_rewards)

    policy_gradient_list = calculate_policy_gradient_element_wise(
        log_probs=log_probs, discounted_rewards=discounted_rewards)

    network.optimizer.zero_grad()
    policy_gradient: torch.Tensor = torch.stack(policy_gradient_list).sum()
    policy_gradient.backward()
    network.optimizer.step()

    if weight_update_mode == 'equation2':
        eq2_params = {
            "discounted_rewards": discounted_rewards,
            "probs": probs,
            "actions": actions,
            "hidden_activities": hidden_activities
        }
        update_output_layer_with_equation2(network, **eq2_params)
    elif weight_update_mode == 'equation4':
        el_params = {
            "probs": probs,
            "hidden_activities": hidden_activities,
            "actions": actions,
        }
        update_output_layer_with_equation4(network, rewards, el_params)
    elif weight_update_mode == 'autograd':
        pass # update done above
    else:
        raise NotImplementedError


def update_output_layer_with_equation2(network: Network, discounted_rewards: torch.Tensor, probs,
                                       actions, hidden_activities):
    """ manual update of output weights and biases"""
    with torch.no_grad(): # to prevent interference with backward
        lr = network.learning_rate
        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
            updates = compute_weight_bias_updates_equation2(actions, idx_action,
                                                            discounted_rewards, probs, hidden_activities)
            weight_updates = updates[0][:-1]
            bias_updates = updates[0][-1]
            weight_vector -= lr* weight_updates
            bias -= lr*bias_updates


def compute_weight_bias_updates_equation2(actions, idx_action, discounted_rewards,
                                          probs, hidden_activities):

    assert len(actions) == len(discounted_rewards)
    assert len(actions) == len(probs)
    assert len(actions) == len(hidden_activities)

    updates = torch.zeros(len(hidden_activities[0]))
    bias_updates = torch.zeros(1).squeeze()
    for idx_time, action in enumerate(actions):
        if idx_action == action:
            kroenecker = 1
        else:
            kroenecker = 0
        updates += (discounted_rewards[idx_time] * (kroenecker - probs[idx_time][:, idx_action]) * hidden_activities[idx_time])
        #bias_updates += (discounted_rewards[idx_time] * (kroenecker - probs[idx_time][:, idx_action])).squeeze()
    return -updates, #-bias_updates


def update_output_layer_with_equation4(network: Network, rewards: List[float],
                                 el_params):

    # update output weights
    with torch.no_grad():
        lr = network.learning_rate
        el_traces = calc_el_traces(**el_params)
        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):

            weight_updates, bias_updates = compute_weight_bias_updates_equation4(rewards, el_traces[idx_action])
            weight_vector -= lr* weight_updates
            bias -= lr*bias_updates


def calc_el_traces(probs, hidden_activities, actions, gamma=0.9):

    n_hidden = hidden_activities[0].size(0)
    n_outputs = probs[0].size(1)
    n_timesteps = len(probs)

    el_traces = torch.zeros([n_outputs, n_hidden, n_timesteps])
    # for t=0
    for idx_action in range(n_outputs):
        el_traces[idx_action, :, 0] = _calc_el_elements(actions[0], idx_action, probs[0], hidden_activities[0])

    for idx_time in range(1,n_timesteps):
        for idx_action in range(n_outputs):
            el_traces[idx_action, :, idx_time] = gamma*el_traces[idx_action, :, idx_time-1] + \
                                                  _calc_el_elements(actions[idx_time], idx_action,
                                                                    probs[idx_time],
                                                                    hidden_activities[idx_time])
    return el_traces


def _calc_el_elements(action, idx_action, probs, hidden_activities):
    if idx_action == action:
        return (1-probs[:,idx_action])*hidden_activities
    else:
        return (0-probs[:,idx_action])*hidden_activities


def compute_weight_bias_updates_equation4(rewards, el_traces):

    weight_updates = torch.zeros(len(el_traces[:,0])-1)
    bias_updates = torch.zeros(1).squeeze()

    for idx_time in range(len(rewards)):
        weight_updates += rewards[idx_time] * el_traces[:-1, idx_time]
        bias_updates += rewards[idx_time] * el_traces[-1,idx_time]
    return -weight_updates, -bias_updates


def update_el_traces(el_traces, prob, hidden_activities, action, gamma=0.9):
    el_traces = gamma * el_traces
    current_el_traces = torch.zeros(el_traces.shape)
    for idx_action in range(el_traces.size(0)):
        current_el_traces[idx_action,:] = _calc_el_elements(action, idx_action, prob,
                                                            hidden_activities)

    el_traces += current_el_traces
    return el_traces


def update_weights_online(network, reward,  el_traces, log_prob, discounted_reward):
    policy_gradient = log_prob * discounted_reward

    network.optimizer.zero_grad()
    policy_gradient.backward()
    network.optimizer.step()

    # update output weights
    with torch.no_grad():
        for idx_action, weight_vector in enumerate(network.output_layer.weight):
            updates = reward*el_traces[idx_action,:]
            weight_vector += network.learning_rate * updates


