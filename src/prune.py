"""Iterative magnitude pruning (IMP) with rewinding.

Frankle & Carbin protocol (adapted to 8x8 digits):
  save w0 (first numbers) -> train -> prune smallest 20% per layer
  -> rewind survivors to w0 -> repeat.

Prune rate: 0.20 per round for W1,W2; 0.10 for output W3 (Frankle App. G:
outputs pruned at half rate). One-shot and magnitude-at-init shortcuts
included for comparison.
"""
import numpy as np


def prune_layerwise(Ws, masks, rate=0.20, out_rate=0.10):
    new_masks = [m.copy() for m in masks]
    for li, (W, M) in enumerate(zip(Ws, masks)):
        r = out_rate if li == 2 else rate
        alive_idx = np.argwhere(M > 0)
        k = int(len(alive_idx) * r)
        if k <= 0:
            continue
        mags = np.abs(W[M > 0])
        # indices of k smallest among alive
        order = np.argpartition(mags, k-1)[:k] if k < len(mags) else np.arange(len(mags))
        kill = alive_idx[order]
        for (i, j) in kill:
            new_masks[li][i, j] = 0.0
    return new_masks


def prune_global(Ws, masks, rate=0.20):
    alive = [(li, i, j, abs(Ws[li][i, j]))
             for li in range(3) for (i, j) in np.argwhere(masks[li] > 0)]
    k = int(len(alive) * rate)
    alive.sort(key=lambda t: t[3])
    new_masks = [m.copy() for m in masks]
    for (li, i, j, _) in alive[:k]:
        new_masks[li][i, j] = 0.0
    return new_masks


def random_mask_like(masks, rng):
    """Random mask with same per-layer sparsity."""
    out = []
    for M in masks:
        keep = int(M.sum())
        total = M.size
        flat = np.zeros(total)
        flat[rng.choice(total, size=keep, replace=False)] = 1.0
        out.append(flat.reshape(M.shape))
    return out


def mask_from_magnitude_at_init(W0, sparsity_per_layer):
    """Keep largest |w0| per layer to reach target sparsity (no training)."""
    out = []
    for W, keep_frac in zip(W0, sparsity_per_layer):
        flat = np.abs(W).ravel()
        k = max(1, int(len(flat) * keep_frac))
        thr = np.sort(flat)[-k]
        out.append((np.abs(W) >= thr).astype(float))
    return out
