import numpy as np
from src.model import init_weights, forward, accuracy, N_PARAMS
from src.data import fixed_split


def test_counts():
    assert N_PARAMS == 50200
    rng = np.random.default_rng(0)
    Ws = init_weights(rng)
    assert Ws[0].shape == (64, 300) and Ws[1].shape == (300, 100) and Ws[2].shape == (100, 10)


def test_forward():
    (Xtr, ytr), _, _ = fixed_split()
    rng = np.random.default_rng(0)
    Ws = init_weights(rng)
    logits, _ = forward(Xtr[:60], Ws, None)
    assert logits.shape == (60, 10)
