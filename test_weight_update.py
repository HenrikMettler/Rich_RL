import torch
import numpy as np

from network import Network, update_with_policy, compute_weight_update_equation2, calculate_discounted_reward


def test_comparison_torch_eq2():
    params= {
        "n_inputs": 2,
        "n_hidden": 4,
        "n_outputs": 3,
        "learning_rate": 0.01
    }
    rng = np.random.default_rng(seed=1234)
    torch.manual_seed(1234)

    state = np.array([0.1, 0.5])
    net = Network(**params, use_autograd_for_output=True)
    action, log_prob, prob, hidden_activities = net.get_action(state, rng)
    reward = 0.75
    updates_all = []

    discounted_rewards = calculate_discounted_reward([reward])
    for idx_action, weight_vector in enumerate(net.output_layer.weight):
        updates = compute_weight_update_equation2(weight_vector, [action], idx_action,
                                                  discounted_rewards, [prob], [hidden_activities])
        updates_all.append(updates.clone())

    update_with_policy(network=net, rewards=[reward], log_probs=[log_prob],
                       use_autograd_for_output=True, actions=[action],
                       probs=[prob], hidden_activities=[hidden_activities])

    for idx_action, weight_vector in enumerate(net.output_layer.weight):
        assert weight_vector._grad == updates_all[idx_action]
