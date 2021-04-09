import torch
import numpy as np
import pytest
from network import normalize_discounted_rewards, calculate_discounted_reward


def test_calculate_discounted_reward():
    gamma = 0.9

    rewards = [0.]
    assert calculate_discounted_reward(rewards, gamma).numpy() == pytest.approx(np.array([0.]))

    rewards = [0.,1.]
    assert calculate_discounted_reward(rewards, gamma) == pytest.approx(np.array([gamma, 1.]))

    rewards = [0., 0. , 1.]
    assert calculate_discounted_reward(rewards, gamma) == pytest.approx(np.array([gamma**2, gamma, 1.]))

    rewards = [0.25, 0.5, 1.]
    expected_rewards = np.array([0.25+ 0.5*gamma + gamma**2, 0.5 + gamma, 1.])
    assert calculate_discounted_reward(rewards, gamma) == pytest.approx(expected_rewards)


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


