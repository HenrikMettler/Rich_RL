import numpy as np
from pytest import fixture


@fixture
def seed():
    return 9876543210


@fixture
def network_params():
    return {"n_inputs": 2, "n_hidden": 4, "n_outputs": 3, "learning_rate_inp2hid": 0.01, 'learning_rate_hid2out': 0.02,
            'beta': 1, 'weight_update_mode': 'autograd'}


@fixture
def states():
    return [np.array([0.1, 0.5]), np.array([0.75, 0.25])]


@fixture
def rewards():
    return [0.75, 0.85]


@fixture
def gamma():
    return 0.9