"""MLP 64-300-100-10 from scratch (NumPy only, no biases).

Counts: 64*300=19200 + 300*100=30000 + 100*10=1000 = 50200 connections.
Each connection holds one number (weight). No biases -> exact count match.

Unit: z = sum(x_i * w_i); a = max(0, z) (ReLU). Output: argmax(logits).
Init: He normal N(0, sqrt(2/fan_in)) -> ~50% plus / 50% minus (fair coin).
Forward/backward with masks for pruning: W_eff = W * mask (mask 0/1).
Deleted connection = 0 and stays 0 (gradient masked too).
"""
import numpy as np

SIZES = [64, 300, 100, 10]
N_PARAMS = 64*300 + 300*100 + 100*10  # 50200


def init_weights(rng):
    W1 = rng.normal(0, np.sqrt(2.0/64), size=(64, 300))
    W2 = rng.normal(0, np.sqrt(2.0/300), size=(300, 100))
    W3 = rng.normal(0, np.sqrt(2.0/100), size=(100, 10))
    return [W1, W2, W3]


def init_masks():
    return [np.ones((64, 300)), np.ones((300, 100)), np.ones((100, 10))]


def forward(X, Ws, masks=None):
    if masks is not None:
        W1 = Ws[0]*masks[0]; W2 = Ws[1]*masks[1]; W3 = Ws[2]*masks[2]
    else:
        W1, W2, W3 = Ws
    z1 = X @ W1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2
    a2 = np.maximum(0, z2)
    logits = a2 @ W3
    cache = (X, z1, a1, z2, a2, logits)
    return logits, cache


def softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits, y):
    return float((logits.argmax(axis=1) == y).mean())


def train_step(Xb, yb, Ws, masks, lr):
    """One SGD step, batch 60. Returns loss."""
    logits, (X, z1, a1, z2, a2, lg) = forward(Xb, Ws, masks)
    n = len(Xb)
    probs = softmax(lg)
    # cross-entropy
    loss = -np.log(probs[np.arange(n), yb] + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), yb] -= 1.0
    dlogits /= n
    if masks is not None:
        W1e = Ws[0]*masks[0]; W2e = Ws[1]*masks[1]; W3e = Ws[2]*masks[2]
    else:
        W1e, W2e, W3e = Ws
    dW3 = a2.T @ dlogits
    da2 = dlogits @ W3e.T
    dz2 = da2 * (z2 > 0)
    dW2 = a1.T @ dz2
    da1 = dz2 @ W2e.T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1
    for W, dW, M in zip(Ws, [dW1, dW2, dW3],
                         masks if masks is not None else [None]*3):
        if M is not None:
            W -= lr * dW * M  # frozen zeros stay zero
        else:
            W -= lr * dW
    return float(loss)


def train_loop(X_train, y_train, Ws, masks, steps=1500, batch=60, lr=0.1,
               seed=0, X_val=None, y_val=None, log_every=None):
    rng = np.random.default_rng(seed)
    n = len(X_train)
    val_curve = []
    for s in range(steps):
        idx = rng.integers(0, n, size=batch)
        train_step(X_train[idx], y_train[idx], Ws, masks, lr)
        if X_val is not None and log_every and (s+1) % log_every == 0:
            logits, _ = forward(X_val, Ws, masks)
            val_curve.append((s+1, accuracy(logits, y_val)))
    return val_curve


def evaluate(X, y, Ws, masks=None):
    logits, _ = forward(X, Ws, masks)
    return accuracy(logits, y)


def count_remaining(masks):
    return int(sum(m.sum() for m in masks))


def dead_units_layer1(Ws, masks):
    """Units in first hidden layer (300) with a way in AND way out,
    dead = ReLU never fires on non-negative inputs.
    A unit with all incoming eff weights <=0 and at least one in-edge
    is dead (since X>=0 -> sum<=0 -> ReLU 0). Single-connection minus => dead.
    Returns (n_alive_path, n_dead, details)."""
    W1e = Ws[0]*masks[0]
    W2e = Ws[1]*masks[1]
    n_in = (masks[0].sum(axis=0) > 0)   # 300 units with way in
    n_out = (masks[1].sum(axis=0) > 0) if False else (masks[1].sum(axis=1) > 0)
    # masks[1] shape (300,100): rows=from layer1 units. sum over cols -> out-degree
    has_path = n_in & n_out
    dead = np.zeros(300, dtype=bool)
    for j in range(300):
        if not has_path[j]:
            continue
        w = W1e[:, j]
        nz = w[masks[0][:, j] > 0]
        if len(nz) == 0:
            continue
        if (nz <= 0).all():
            dead[j] = True
    return int(has_path.sum()), int((dead & has_path).sum())


def sign_stats(Ws):
    allw = np.concatenate([w.ravel() for w in Ws])
    nz = allw[allw != 0]
    if len(nz) == 0:
        return 0.0
    return float((nz > 0).mean())
