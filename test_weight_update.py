import torch
import numpy as np
import pytest

from network import Network, update_with_policy, compute_weight_update_equation2, calculate_discounted_rewards, normalize_discounted_rewards


def test_comparison_torch_eq2():
    params= {
        "n_inputs": 2,
        "n_hidden": 4,
        "n_outputs": 3,
        "learning_rate": 0.01
    }
    rng = np.random.default_rng(seed=1234)
    torch.manual_seed(1234)

    states = [np.array([0.1, 0.5]), np.array([0.75, 0.25])]
    rewards = [0.75, 0.85]
    net = Network(**params, use_autograd_for_output=True)
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
    updates = torch.empty_like(net.output_layer.weight)
    with torch.no_grad():
        for idx_action, weight_vector in enumerate(net.output_layer.weight):
            updates[idx_action] = compute_weight_update_equation2(weight_vector, actions, idx_action,
                                                  discounted_rewards, probs, hidden_activities_all)

    update_with_policy(network=net, rewards=rewards, log_probs=log_probs,
                       use_autograd_for_output=True, actions=actions,
                       probs=probs, hidden_activities=probs)

    assert updates.numpy() != pytest.approx(0.)
    assert net.output_layer.weight._grad.detach().numpy() == pytest.approx(updates.numpy())
