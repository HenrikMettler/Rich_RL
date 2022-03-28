import torch
import numpy as np

from network import Network
from typing import AnyStr, List


# todo: ask J: is **_ a good solution to ignore seeds?
def run_curriculum(env, net, rule, max_n_alterations, n_alterations_per_new_env, prob_alteration_dict, spatial_novelty_distance_decay,
                   n_episodes_per_alteration, n_steps_max, temporal_novelty_decay, spatial_novelty_time_decay, rng, **_):

    rewards_over_alterations: List[float] = []

    for _ in range(1, max_n_alterations):

        env.alter_env(n_alterations=n_alterations_per_new_env, prob_alteration_dict=prob_alteration_dict,
                      spatial_novelty_distance_decay=spatial_novelty_distance_decay)
        rewards_over_episodes = play_episodes(env=env, net=net, rule=rule,
                                              n_episodes=n_episodes_per_alteration, n_steps_max=n_steps_max,
                                              temporal_novelty_decay=temporal_novelty_decay,
                                              spatial_novelty_time_decay=spatial_novelty_time_decay,
                                              rng=rng)

        rewards_over_alterations.append(np.mean(rewards_over_episodes))

    return rewards_over_alterations


def play_episodes(env, net, rule, n_episodes, n_steps_max, temporal_novelty_decay, spatial_novelty_time_decay, rng):

    rewards_over_episodes = []
    temporal_novelty = 1
    # runs
    for episode in range(n_episodes):
        reward = play_episode(env, net, rule, n_steps_max, temporal_novelty, rng)
        rewards_over_episodes.append(reward)
        temporal_novelty *= temporal_novelty_decay
        env.spatial_novelty_grid_time_decay(spatial_novelty_time_decay)
    return rewards_over_episodes


def play_episode(env, net, rule, n_steps_max, temporal_novelty, rng):

    state = env.respawn()["image"][:,:,0].flatten()
    log_probs: List[torch.Tensor] = []
    probs: List[float] = []
    actions: List[int] = []
    hidden_activities_all = []
    rewards: List[float] = []
    spatial_novelty_signals = []

    for steps in range(n_steps_max):

        action, prob, hidden_activities = net.get_action(state, rng)

        hidden_activities = torch.cat((hidden_activities, torch.ones(1)), 0)
        log_prob = torch.log(prob.squeeze(0)[action])

        new_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        rewards.append(reward)
        hidden_activities_all.append(hidden_activities)
        spatial_novelty_currently = env.spatial_novelty_grid[env.agent_pos[0], env.agent_pos[1]]
        spatial_novelty_signals.append(spatial_novelty_currently)

        if done or steps == n_steps_max -1:
            update_params = {
                "rewards": rewards,
                "probs": probs,
                "log_probs": log_probs,
                "actions": actions,
                "hidden_activities": hidden_activities_all,
                'temporal_novelty': temporal_novelty,
                'spatial_novelty': spatial_novelty_signals
            }
            update_weights_offline(network=net, weight_update_mode='evolved-rule',
                                    normalize_discounted_rewards_b=False,
                                    rule=rule, **update_params)

            break

        state = new_state[:,:,0].flatten()
    return np.sum(rewards)


# def update_inp_hidden_weights_offline(net, rewards, log_probs, normalize_discounted_rewards_b=False ):
#
#     discounted_rewards = calculate_discounted_rewards(rewards)
#
#     if normalize_discounted_rewards_b:
#         discounted_rewards = normalize_discounted_rewards(discounted_rewards)
#
#     policy_gradient_list = calculate_policy_gradient_element_wise(
#         log_probs=log_probs, discounted_rewards=discounted_rewards)
#
#     torch.autograd.set_detect_anomaly(True)
#     net.optimizer.zero_grad()
#     policy_gradient: torch.Tensor = torch.stack(policy_gradient_list).sum()
#     policy_gradient.backward(retain_graph=True)
#     net.optimizer.step()


# def update_output_weights_online_with_rule(rule, net, reward, el_traces, temporal_novelty, spatial_novelty):
#
#     # expand scalar values in dimensionality of hidden_layer (incl bias)
#     n = net.output_layer.in_features + 1
#     reward_expanded = reward*torch.ones(n)
#     temporal_novelty_expanded = temporal_novelty*torch.ones(n)
#     spatial_novelty_expanded = spatial_novelty*torch.ones(n)
#
#     with torch.no_grad():
#         lr = net.learning_rate_hid2out
#         for idx_action, (weight_vector, bias) in enumerate(zip(net.output_layer.weight, net.output_layer.bias)):
#             updates = rule(torch.stack([reward_expanded, el_traces[idx_action,:],
#                                         temporal_novelty_expanded, spatial_novelty_expanded], 1))
#
#             weight_update = updates[:-1].squeeze()
#             bias_update = updates[-1].squeeze()
#             weight_vector -= lr * weight_update
#             bias -= lr * bias_update


def calculate_discounted_rewards(rewards: List[float], gamma=0.9):
    rewards = torch.Tensor(rewards)
    n = len(rewards)
    discounted_rewards = torch.empty(n)
    for t in range(n):
        gamma_factor = gamma**torch.arange(n-t, dtype=torch.float)
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


def update_weights_offline(network: Network, rewards, log_probs, probs, actions, hidden_activities,
                   temporal_novelty=None,
                   spatial_novelty=None,
                   weight_update_mode: AnyStr = 'autograd',
                   normalize_discounted_rewards_b = True,
                   rule = None):


    discounted_rewards = calculate_discounted_rewards(rewards)
    # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
    if normalize_discounted_rewards_b:
        discounted_rewards = normalize_discounted_rewards(discounted_rewards)

    policy_gradient_list = calculate_policy_gradient_element_wise(
        log_probs=log_probs, discounted_rewards=discounted_rewards)

    network.optimizer.zero_grad()
    policy_gradient: torch.Tensor = torch.stack(policy_gradient_list).sum()
    policy_gradient.backward()  # this doesn't update the hidden to output weights, since they have require_grad=False
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

    elif weight_update_mode == 'evolved-rule':
        if rule == None:
            raise ValueError('rule must be defined for update with rule')
        el_params = {
            "probs": probs,
            "hidden_activities": hidden_activities,
            "actions": actions,
        }
        el_traces = calc_el_traces(**el_params)
        update_output_layer_with_evolved_rule_offline(rule=rule, network=network, rewards=rewards, el_traces=el_traces,
                                                      temporal_novelty=temporal_novelty, spatial_novelty=spatial_novelty)

    elif weight_update_mode == 'autograd':
        pass  # update done above
    else:
        raise NotImplementedError


def update_output_layer_with_evolved_rule_offline(rule, network, rewards, el_traces, temporal_novelty, spatial_novelty):
    # update output weights
    with torch.no_grad():
        lr = network.learning_rate_hid2out
        rewards_expanded = _expand_signal_in_hidden_layer_dim(signal=rewards,
                                                              hidden_layer_dim=network.output_layer.weight.size(1) + 1)
        if spatial_novelty is not None:
            spatial_novelty_expanded = _expand_signal_in_hidden_layer_dim(signal=spatial_novelty,
                                                        hidden_layer_dim=network.output_layer.weight.size(1) + 1)
        else:
            spatial_novelty_expanded = None
        if temporal_novelty is not None:
            temporal_novelty_expanded = temporal_novelty * torch.ones([network.output_layer.weight.size(1) + 1, len(rewards)])
        else:
            temporal_novelty_expanded = None

        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):

            weight_updates, bias_updates = compute_weight_bias_updates_with_rule_offline\
                (rewards_expanded, el_traces[idx_action], rule, temporal_novelty_expanded, spatial_novelty_expanded)
            weight_vector += lr* weight_updates
            bias += lr*bias_updates


def _expand_signal_in_hidden_layer_dim(signal, hidden_layer_dim):
    ones = torch.ones(hidden_layer_dim)
    ones = ones.resize_((len(ones), 1))
    signal_t = torch.tensor(signal)
    signal_t.resize_(1, len(signal))
    signal_torch_expanded = ones * signal_t
    return signal_torch_expanded


def compute_weight_bias_updates_with_rule_offline(rewards, el_traces, rule, temporal_novelty=None, spatial_novelty=None):
    weight_updates = torch.zeros(len(el_traces[:, 0]) - 1)
    bias_updates = torch.zeros(1).squeeze()

    for idx_time in range(rewards.shape[1]):
        if temporal_novelty is None and spatial_novelty is None:
            updates = rule(torch.stack([rewards[:,idx_time], el_traces[:, idx_time]],1))
        elif spatial_novelty is None:
            updates = rule(torch.stack([rewards[:,idx_time], el_traces[:, idx_time], temporal_novelty[:, idx_time]],1))
        elif temporal_novelty is None:
            updates = rule(torch.stack([rewards[:,idx_time], el_traces[:, idx_time], spatial_novelty[:, idx_time]],1))
        else:
            updates = rule(torch.stack([rewards[:,idx_time], el_traces[:, idx_time], temporal_novelty[:, idx_time],
                                        spatial_novelty[:, idx_time]],1))
        weight_updates += updates[:-1].squeeze()
        bias_updates += updates[-1].squeeze()
    return weight_updates, bias_updates


def update_output_layer_with_equation2(network: Network, discounted_rewards: torch.Tensor, probs,
                                       actions, hidden_activities):
    """ manual update of output weights and biases"""
    with torch.no_grad(): # to prevent interference with backward
        lr = network.learning_rate_hid2out
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
        lr = network.learning_rate_hid2out
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


def update_weights_online_with_policy_gradient(network, reward,  el_traces, log_prob, discounted_reward):
    policy_gradient = -log_prob * discounted_reward

    network.optimizer.zero_grad()
    policy_gradient.backward()
    network.optimizer.step()

    # update output weights
    with torch.no_grad():
        lr = network.learning_rate_hid2out
        for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
            updates = compute_weight_bias_update_online(reward, el_traces[idx_action,:])
            weight_update = updates[:-1]
            bias_update = updates[-1]
            weight_vector -= lr * weight_update
            bias -= lr * bias_update


def compute_weight_bias_update_online(reward, el_traces_per_output):
    return -reward*el_traces_per_output


# def update_weights_online_with_rule(rule, network, reward,  el_traces, log_prob, discounted_reward,
#                                     done, expected_cum_reward_per_episode):
#     policy_gradient = -log_prob * discounted_reward
#
#     network.optimizer.zero_grad()
#     policy_gradient.backward()
#     network.optimizer.step()
#
#     # update_output weights
#     with torch.no_grad():
#         lr = network.learning_rate_hid2out
#         rewards_torch_expanded = reward * torch.ones(network.output_layer.weight.size(1)+1) # +1 for bias
#         done_torch_expanded = done * torch.ones_like(rewards_torch_expanded)
#         expected_cum_reward_per_episode_torch_expanded = expected_cum_reward_per_episode * torch.ones_like(
#             rewards_torch_expanded)
#
#         for idx_action, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
#             updates = rule(torch.stack([rewards_torch_expanded, el_traces[idx_action,:],
#                                         done_torch_expanded, expected_cum_reward_per_episode_torch_expanded],1)).squeeze()
#             weight_update = updates[:-1]
#             bias_update = updates[-1]
#             weight_vector += lr * weight_update
#             bias += lr * bias_update


def play_some_episodes_with_trained_agent(data, params):
    env = data['env']
    net = data['network']
    n_episodes = 200
    n_steps_max = 50
    rng = np.random.default_rng(seed=params["seed"])
    states = []
    agent_positions = []
    agent_directions = []
    for episode in range(n_episodes):
        state = env.respawn()["image"].flatten()

        for steps in range(n_steps_max):

            action, prob, _ = net.get_action(state, rng)
            new_state, reward, done, _ = env.step(action)
            states.append(new_state)
            agent_positions.append(env.agent_pos)
            agent_directions.append(env.agent_dir)

            if done or steps == n_steps_max - 1:
                break
            state = new_state.flatten()