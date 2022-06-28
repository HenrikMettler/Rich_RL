import torch
import numpy as np

from network import Network
from typing import AnyStr, List


def run_curriculum(env, net, rule, max_n_alterations, n_alterations_per_new_env, prob_alteration_dict,
                   spatial_novelty_distance_decay, n_episodes_per_alteration, n_steps_max, temporal_novelty_decay,
                   spatial_novelty_time_decay, rng):

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

    state = env.respawn()["image"][:, :, 0].flatten()
    log_probs: List[torch.Tensor] = []
    probs: List[float] = []
    actions: List[int] = []
    hidden_activities_over_time = []
    rewards: List[float] = []
    spatial_novelty_signals = []

    for steps in range(n_steps_max):

        action, prob, hidden_activities = net.get_action(state, rng)
        hidden_activities = torch.cat((hidden_activities, torch.ones(1)))
        log_prob = torch.log(prob[action])

        new_state, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        probs.append(prob)
        actions.append(action)
        rewards.append(reward)
        hidden_activities_over_time.append(hidden_activities)
        spatial_novelty_currently = env.spatial_novelty_grid[env.agent_pos[0], env.agent_pos[1]]
        spatial_novelty_signals.append(spatial_novelty_currently)

        if done or steps == n_steps_max -1:
            update_params = {
                "rewards": torch.Tensor(rewards),
                "probs": torch.stack(probs),
                "log_probs": torch.stack(log_probs),
                "actions": actions,
                "hidden_activities": torch.stack(hidden_activities_over_time),
                'temporal_novelty': temporal_novelty,
                'spatial_novelty': torch.Tensor(spatial_novelty_signals),
            }
            update_weights_offline(network=net, weight_update_mode='evolved-rule', normalize_discounted_rewards_b=False,
                                   rule=rule, **update_params)

            break

        state = new_state[:, :, 0].flatten()
    return np.sum(rewards)


def update_weights_offline(network: Network, rewards, log_probs, probs: torch.Tensor, actions, hidden_activities,
                           temporal_novelty=None, spatial_novelty=None, weight_update_mode: AnyStr = 'autograd',
                           normalize_discounted_rewards_b=False, rule=None):

    discounted_rewards = calculate_discounted_rewards(rewards, norm_disc_r_bool=normalize_discounted_rewards_b)

    # inp 2 hidden updates with PG - only done if lr is not 0
    if abs(network.learning_rate_inp2hid) < 10e-9:
        policy_gradient_element_wise = calculate_policy_gradient_element_wise(log_probs=log_probs,
                                                                      discounted_rewards=discounted_rewards)

        network.optimizer.zero_grad()
        policy_gradient: torch.Tensor = policy_gradient_element_wise.sum()
        policy_gradient.backward()  # this doesn't update the hidden to output weights, since they have require_grad=False
        network.optimizer.step()

    if weight_update_mode == 'evolved-rule':
        if rule is None:
            raise ValueError('rule must be defined for update with rule')

        sampled_action_minus_action_prob_over_time = calculate_sampled_actions_minus_probs_time_array(actions, probs)
        eligibility_over_time = calculate_eligibility_over_time(sampled_action_minus_action_prob_over_time,
                                                                hidden_activities)

        update_output_layer_with_evolved_rule_offline(rule=rule, network=network, discounted_rewards=discounted_rewards,
                                                      eligibility_over_time=eligibility_over_time, temporal_novelty=
                                                      temporal_novelty, spatial_novelty=spatial_novelty)

    elif weight_update_mode == 'equation2':
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


def calculate_discounted_rewards(rewards: torch.Tensor, gamma=0.9, norm_disc_r_bool=True):
    n = len(rewards)
    discounted_rewards = torch.empty(n)
    for t in range(n):
        gamma_factor = gamma**torch.arange(n-t, dtype=torch.float)
        discounted_rewards[t] = gamma_factor @ rewards[t:]

    def _normalize_discounted_rewards(discounted_rewards: torch.Tensor):
        if len(discounted_rewards) == 0:
            raise RuntimeError("Length of discounted rewards must be non-zero")
        elif len(discounted_rewards) == 1:
            return torch.Tensor([0.])
        return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-12)

    if norm_disc_r_bool:
        # normalized discounted rewards according to: https://arxiv.org/abs/1506.02438
        discounted_rewards = _normalize_discounted_rewards(discounted_rewards)
    return discounted_rewards


def calculate_policy_gradient_element_wise(log_probs: torch.Tensor, discounted_rewards: torch.Tensor):
    # policy_gradient_list: List[torch.Tensor] = []
    # for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
    #     # -, since we do gradient ascent on the expected discounted rewards, not descent
    #     policy_gradient_list.append(-log_prob*discounted_reward)
    policy_gradient_element_wise = log_probs * discounted_rewards
    return - policy_gradient_element_wise # -, since gradient ascent on the expected discounted rewards, not descent


def calculate_sampled_actions_minus_probs_time_array(actions: List[int], probs: torch.Tensor):
    assert len(actions) == probs.shape[0]

    kroenecker_delta = torch.zeros(probs.shape, requires_grad=False)
    r = torch.arange(0, len(actions))
    kroenecker_delta[r, actions] = 1.0
    sampled_action_minus_action_prob_over_time = kroenecker_delta - probs

    return sampled_action_minus_action_prob_over_time


def calculate_eligibility_over_time(sampled_action_minus_action_prob_over_time, hidden_activities_over_time):

    assert len(sampled_action_minus_action_prob_over_time) == len(hidden_activities_over_time)

    # time_dim = len(hidden_activities_over_time)
    # n_hidden_with_bias = hidden_activities_over_time.shape[1]
    # n_actions = sampled_action_minus_action_prob_over_time.shape[1]
    # eligibility_over_time = torch.zeros((time_dim, n_actions, n_hidden_with_bias))
    #
    # for idx_time in range(time_dim):
    #     eligibility_over_time[idx_time] = torch.outer(sampled_action_minus_action_prob_over_time[idx_time],
    #                                         hidden_activities_over_time[idx_time])

    # https://pytorch.org/docs/stable/generated/torch.bmm.html
    eligibility_over_time = torch.bmm(sampled_action_minus_action_prob_over_time.unsqueeze(2),
                                      hidden_activities_over_time.unsqueeze(1))
    return eligibility_over_time


def update_output_layer_with_evolved_rule_offline(rule, network, discounted_rewards,
                                                  eligibility_over_time, temporal_novelty,
                                                  spatial_novelty):

    """ Dimensions:
        discounted rewards: T
        eligibility_over_time: T * N_ACT * N_Hidden(+1)
        spatial_novelty: T
        temporal_novelty: 1
    """

    with torch.no_grad():
        lr = network.learning_rate_hid2out

        # todo: can the updates be vectorized over the output dimension?:
        for idx_output, (weight_vector, bias) in enumerate(zip(network.output_layer.weight, network.output_layer.bias)):
            weight_bias_vector = concat_weight_bias(weight_vector, bias)
            eligibility_over_time_current_output_unit = eligibility_over_time[:, idx_output, :]
            weight_updates, bias_updates = \
                compute_updates_per_output_unit_with_rule_offline(rule, discounted_rewards,
                                                                  eligibility_over_time_current_output_unit,
                                                                  temporal_novelty, spatial_novelty, weight_bias_vector)
            weight_vector += lr*weight_updates
            bias += lr*bias_updates


def concat_weight_bias(weight_vector, bias):
    bias_copy = bias.detach().clone()
    bias_copy.resize_(1)
    weight_bias_vector = torch.cat((weight_vector, bias_copy))
    return weight_bias_vector


# todo: test with function (eg dr*e+t-s) instead of rule
def compute_updates_per_output_unit_with_rule_offline(rule, discounted_rewards, eligibility_over_time,
                                                      temporal_novelty=None, spatial_novelty=None,
                                                      weight_bias_vector=None):

    # todo: adapt to case where not all arguments exist
    if spatial_novelty is None or temporal_novelty is None or weight_bias_vector is None:
        raise ValueError('Configuration without all arguments are currently not working')

    hidden_dim = len(weight_bias_vector)
    time_dim = len(discounted_rewards)

    # expand signals which don't have hidden dimensionality and/ or time dimensionality
    discounted_rewards_exp, temporal_novelty_exp, spatial_novelty_exp, weight_bias_vector_exp = \
        _expand_signals(discounted_rewards=discounted_rewards, temporal_novelty=temporal_novelty,
                        spatial_novelty=spatial_novelty, weight_bias_vector=weight_bias_vector)

    # collapse time dimension
    discounted_rewards_coll, eligibility_coll, temporal_novelty_coll, spatial_novelty_coll, weight_bias_vector_coll = \
        _collapse_along_hidden_dimension(discounted_rewards_exp, eligibility_over_time, temporal_novelty_exp,
                                         spatial_novelty_exp, weight_bias_vector_exp)

    rule_input = torch.stack([discounted_rewards_coll, eligibility_coll, temporal_novelty_coll, spatial_novelty_coll,
         weight_bias_vector_coll], 1)
    all_updates = rule(rule_input)
    update_per_time_step = torch.reshape(all_updates, (time_dim, hidden_dim))  # todo: check whether this does the right thing dimension wise!
    update_per_weight = torch.sum(update_per_time_step, dim=0)

    weight_updates = update_per_weight[:-1]
    bias_update = update_per_weight[-1]
    return weight_updates, bias_update


# todo: is it good practise to have functions with that many output arguments? (discuss with J)
def _expand_signals(discounted_rewards, temporal_novelty, spatial_novelty, weight_bias_vector):

    hidden_dim = len(weight_bias_vector)
    time_dim = len(discounted_rewards)

    # expand discounted rewards and spatial novelty along hidden dimension
    discounted_rewards_exp = torch.outer(discounted_rewards, torch.ones(hidden_dim))
    spatial_novelty_exp = torch.outer(spatial_novelty, torch.ones(hidden_dim))

    # expand temporal novelty along time and hidden dimension
    temporal_novelty_exp = temporal_novelty * torch.ones([time_dim, hidden_dim])

    # expand weight_bias_vector along time dimension
    weight_bias_vector_exp = torch.outer(torch.ones(time_dim), weight_bias_vector)

    return discounted_rewards_exp, temporal_novelty_exp, spatial_novelty_exp, weight_bias_vector_exp


def _collapse_along_hidden_dimension(discounted_rewards_exp, eligibility_over_time, temporal_novelty_exp,
                                        spatial_novelty_exp, weight_bias_vector_exp):

    discounted_rewards_coll = torch.reshape(discounted_rewards_exp, (-1,))
    eligibility_coll = torch.reshape(eligibility_over_time, (-1,))
    temporal_novelty_coll = torch.reshape(temporal_novelty_exp, (-1,))
    spatial_novelty_coll = torch.reshape(spatial_novelty_exp, (-1,))
    weight_bias_vector_coll = torch.reshape(weight_bias_vector_exp, (-1,))

    return discounted_rewards_coll, eligibility_coll, temporal_novelty_coll, spatial_novelty_coll, \
           weight_bias_vector_coll


def update_output_layer_with_equation2(network: Network, discounted_rewards: torch.Tensor, probs,
                                       actions, hidden_activities):
    """ manual update of output weights and biases"""
    with torch.no_grad():  # to prevent interference with backward
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

    return -updates_vector


def update_output_layer_with_equation4(network: Network, rewards: List[float], el_params):

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
    n_outputs = len(probs[0])
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