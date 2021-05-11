import numpy as np
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE
seed = 1234
rng = np.random.default_rng(seed=seed)
n_operators = 6 # Mul, add, sub, const float, const 0.5, const 2

onlinelr_dna_20internal = [
    ID_INPUT_NODE,  # r (x_0)
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
    ID_INPUT_NODE,  # el_trace (x_1)
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
    ID_INPUT_NODE,  # episode_done (boolean) (x_2)
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
    0,  # Mul ->   (address 3)
    0,  # r
    1,  # el_trace
    rng.integers(low=0,high=n_operators),  # Sub -> x_0^2 - x_1 (address 4)
    rng.integers(low=0,high=4),
    rng.integers(low=0,high=4),
    rng.integers(low=0,high=n_operators),  # (address 5)
    rng.integers(low=0,high=5),
    rng.integers(low=0,high=5),
    rng.integers(low=0,high=n_operators),  # (address 6)
    rng.integers(low=0,high=6),
    rng.integers(low=0,high=6),
    rng.integers(low=0,high=n_operators),  # (address 7)
    rng.integers(low=0,high=7),
    rng.integers(low=0,high=7),
    rng.integers(low=0,high=n_operators),  # (address 8)
    rng.integers(low=0,high=8),
    rng.integers(low=0,high=8),
    rng.integers(low=0,high=n_operators),  # (address 9)
    rng.integers(low=0,high=9),
    rng.integers(low=0,high=9),
    rng.integers(low=0,high=n_operators),  # (address 10)
    rng.integers(low=0,high=10),
    rng.integers(low=0,high=10),
    rng.integers(low=0,high=n_operators),  # (address 11)
    rng.integers(low=0,high=11),
    rng.integers(low=0,high=11),
    rng.integers(low=0,high=n_operators),  # (address 12)
    rng.integers(low=0,high=12),
    rng.integers(low=0,high=12),
    rng.integers(low=0,high=n_operators),  # (address 13)
    rng.integers(low=0,high=13),
    rng.integers(low=0,high=13),
    rng.integers(low=0,high=n_operators),  # (address 14)
    rng.integers(low=0,high=14),
    rng.integers(low=0,high=14),
    rng.integers(low=0,high=n_operators),  # (address 15)
    rng.integers(low=0,high=15),
    rng.integers(low=0,high=15),
    rng.integers(low=0, high=n_operators),  # (address 16)
    rng.integers(low=0,high=16),
    rng.integers(low=0,high=16),
    rng.integers(low=0, high=n_operators),  # (address 17)
    rng.integers(low=0,high=17),
    rng.integers(low=0,high=17),
    rng.integers(low=0, high=n_operators),  # (address 18)
    rng.integers(low=0,high=18),
    rng.integers(low=0,high=18),
    rng.integers(low=0, high=n_operators),  # (address 19)
    rng.integers(low=0,high=19),
    rng.integers(low=0,high=19),
    rng.integers(low=0, high=n_operators),  # (address 20)
    rng.integers(low=0,high=20),
    rng.integers(low=0,high=20),
    rng.integers(low=0, high=n_operators),  # (address 21)
    rng.integers(low=0, high=21),
    rng.integers(low=0, high=21),
    rng.integers(low=0, high=n_operators),  # (address 22)
    rng.integers(low=0, high=22),
    rng.integers(low=0, high=22),
    ID_OUTPUT_NODE,
    3,
    ID_NON_CODING_GENE,
]