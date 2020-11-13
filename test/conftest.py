import numpy as np
from pytest import fixture

@fixture
def rng_seed():
    return 1234