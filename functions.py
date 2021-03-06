import torch
import numpy as np

from network import Network
from typing import AnyStr, List

from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE


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
                   weight_update_mode: AnyStr = 'autograd',
                   normalize_discounted_rewards_b = True):
    """adapted from: https://medium.com/@thechrisyoon/
    deriving-policy-gradients-and-implementing-reinforce-f887949bd63"""

    discounted_rewards = calculate_discounted_rewards(rewards)
    # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
    if normalize_discounted_rewards_b:
        discounted_rewards = normalize_discounted_rewards(discounted_rewards)

    log_probs_t = torch.Tensor(log_probs)
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
        pass  # update done above
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

    kroenecker_vector = torch.Tensor([1 if action == idx_action else 0 for action in actions])
    prob_vector = torch.Tensor([prob[:, idx_action] for prob in probs])
    parenthesis = kroenecker_vector - prob_vector
    left_term = discounted_rewards * parenthesis
    hidden_activities_t = torch.stack(hidden_activities)
    updates_vector = torch.matmul(left_term, hidden_activities_t)

    return -updates_vector, ##-bias_updates


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
    kroenecker = torch.Tensor([1 if idx_action == actions[0] else 0 for idx_action in range(n_outputs)])
    pre_factor = kroenecker - probs[0]
    el_traces[:,:,0] = torch.matmul(pre_factor.T, hidden_activities[0].unsqueeze(0))

    for idx_time in range(1,n_timesteps):
        kroenecker = torch.Tensor([1 if idx_action == actions[idx_time] else 0 for idx_action in range(n_outputs)])
        pre_factor = kroenecker - probs[idx_time]
        el_update = torch.matmul(pre_factor.T, hidden_activities[idx_time].unsqueeze(0))
        el_traces[:,:,idx_time] = gamma * el_traces[:,:,idx_time-1] + el_update

    return el_traces


def compute_weight_bias_updates_equation4(rewards, el_traces):

    weight_updates = torch.zeros(len(el_traces[:,0])-1)
    bias_updates = torch.zeros(1).squeeze()

    for idx_time in range(len(rewards)):
        weight_updates += rewards[idx_time] * el_traces[:-1, idx_time]
        bias_updates += rewards[idx_time] * el_traces[-1,idx_time]
    return -weight_updates, -bias_updates


def update_el_traces(el_traces, probs, hidden_activities, action, gamma=0.9):
    el_traces = gamma * el_traces

    n_outputs = el_traces.size(0)

    kroenecker = torch.Tensor([1 if idx_action == action else 0 for idx_action in range(n_outputs)])
    pre_factor = kroenecker - probs
    current_el_traces = torch.matmul(pre_factor.T, hidden_activities.unsqueeze(0))

    el_traces += current_el_traces
    return el_traces


def update_weights_online(network, reward,  el_traces, log_prob, discounted_reward):
    policy_gradient = -log_prob * discounted_reward

    network.optimizer.zero_grad()
    policy_gradient.backward()
    network.optimizer.step()

    # update output weights
    with torch.no_grad():
        lr = network.learning_rate
        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
            updates = compute_weight_bias_update_online(reward, el_traces[idx_action,:])
            weight_update = updates[:-1]
            bias_update = updates[-1]
            weight_vector -= lr * weight_update
            bias -= lr * bias_update


def compute_weight_bias_update_online(reward, el_traces_per_output):
    return -reward*el_traces_per_output


def update_weights_online_with_rule(rule, network, reward,  el_traces, log_prob, discounted_reward,
                                    done, expected_cum_reward_per_episode):
    policy_gradient = -log_prob * discounted_reward

    network.optimizer.zero_grad()
    policy_gradient.backward()
    network.optimizer.step()

    # update_output weights
    with torch.no_grad():
        lr = network.learning_rate
        rewards_torch_expanded = reward * torch.ones(network.output_layer.weight.size(1)+1) # +1 for bias
        done_torch_expanded = done * torch.ones_like(rewards_torch_expanded)
        expected_cum_reward_per_episode_torch_expanded = expected_cum_reward_per_episode * torch.ones_like(
            rewards_torch_expanded)

        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
        # todo: check if possible to set updates values into weight_vector to save them for update at next time step
            updates = rule(torch.stack([rewards_torch_expanded, el_traces[idx_action,:],
                                        done_torch_expanded, expected_cum_reward_per_episode_torch_expanded],1)).squeeze()
            weight_update = updates[:-1]
            bias_update = updates[-1]
            weight_vector += lr * weight_update
            bias += lr * bias_update


def update_weights_with_rule():
    raise NotImplementedError


def initialize_genome_with_rxet_prior(n_inputs, n_hidden, n_operators, max_arity, rng):
    max_arity += 1 # account for operator gene not counted in arity
    dna = []
    # add inputs
    for idx in range(n_inputs*max_arity):
        if idx%max_arity == 0:
            dna.append(ID_INPUT_NODE)
        else:
            dna.append(ID_NON_CODING_GENE)

    # add r*et prior
    dna.append(0)  # Assuming Mul is the first operator
    dna.append(0)  # Assuming r is the first input
    dna.append(1)  # Assuming et is the second input

    # add random n_hidden-1 genes
    for idx in range(max_arity,n_hidden*max_arity):  # start at max_arity because 1st gene is set
        if idx%max_arity == 0:
            dna.append(rng.integers(low=0,high=n_operators))  # (address 4)
        else:
            dna.append(rng.integers(low=0,high=int(idx/max_arity)+n_inputs))

    # add output
    dna.append(ID_OUTPUT_NODE)
    dna.append(n_inputs) # r*et has node-index of n_inputs
    dna.append(ID_NON_CODING_GENE)

    return dna