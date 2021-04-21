import torch
import numpy as np
import pytest
import sys

sys.path.insert(0, '../')
from functions import normalize_discounted_rewards, calculate_discounted_rewards, \
    calculate_policy_gradient_element_wise


def test_calculate_discounted_rewards(gamma):

    rewards = [0.]
    assert calculate_discounted_rewards(rewards, gamma).numpy() == pytest.approx(np.array([0.]))

    rewards = [0.,1.]
    assert calculate_discounted_rewards(rewards, gamma) == pytest.approx(np.array([gamma, 1.]))

    rewards = [0., 0. , 1.]
    assert calculate_discounted_rewards(rewards, gamma) == pytest.approx(np.array([gamma**2, gamma, 1.]))

    rewards = [0.25, 0.5, 1.]
    expected_rewards = np.array([0.25+ 0.5*gamma + gamma**2, 0.5 + gamma, 1.])
    assert calculate_discounted_rewards(rewards, gamma) == pytest.approx(expected_rewards)


def test_normalize_discounted_rewards():
    discounted_rewards = torch.Tensor([])
    with pytest.raises(RuntimeError):
        normalize_discounted_rewards(discounted_rewards)

    discounted_rewards = torch.Tensor([0.75])
    assert normalize_discounted_rewards(discounted_rewards) == torch.zeros_like(discounted_rewards)

    discounted_rewards = torch.Tensor([0.75, 0.75])
    assert torch.all(normalize_discounted_rewards(discounted_rewards) == torch.zeros_like(discounted_rewards))

    discounted_rewards = torch.Tensor([0.75, 0.85])
    assert normalize_discounted_rewards(discounted_rewards).numpy() == pytest.approx(np.array([-1/np.sqrt(2),1/np.sqrt(2)]))


def test_comp_policy_gradient_elwise_vectorwise(gamma):
    rewards = [0.5, 1, 2, 100]
    probs = [0.1, 0.5, 0.8, 0.9]
    log_probs = torch.log(torch.tensor(probs))
    discounted_rewards = calculate_discounted_rewards(rewards, gamma)

    policy_gradient_list = calculate_policy_gradient_element_wise(log_probs, discounted_rewards)
    policy_gradient_from_elwise  = torch.stack(policy_gradient_list).sum()

    policy_gradient_vectorwise = torch.sum(-log_probs.T * discounted_rewards)

    assert policy_gradient_from_elwise.numpy() == pytest.approx(policy_gradient_vectorwise.numpy())


#def test_calc_el_traces():
#    raise NotImplementedError