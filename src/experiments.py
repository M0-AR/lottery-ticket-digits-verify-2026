"""Full experiment suite: dense, IMP search, ticket vs ablations.

Covers every claim in the video:
 D0 dense baseline (50200 conns) -> ~98.2%
 P1 IMP rounds: 7 (~20% left), 13 (~6%), 18 (~1037 conns, 2%), 22 (~462, <1%)
 P2 one-shot vs iterative (iterative +~6%)
 P3 magnitude-at-init shortcut (~79%)
 P4 small-dense 64-14-10 (1036 conns) baseline
 T1 ticket (rewind to w0) vs new-start (reinit) at 1037 and 462
 T2 random-mask control
 T3 sign ablations: keep-sign/new-size, same-size, keep-size/new-sign, flip-all
 T4 longer training 6000 steps, bigger LR 4x
 T5 untrained accuracy (ticket ~77% vs guess ~11%)
 H  hidden pattern: dead ReLU units, 86% plus-sign bias, sign preservation 99%
"""
import copy, json, time
import numpy as np
from .data import fixed_split
from .model import (init_weights, init_masks, train_loop, evaluate,
                    count_remaining, dead_units_layer1, sign_stats, forward)
from .prune import prune_layerwise, random_mask_like, mask_from_magnitude_at_init


def run_dense(seed_run, steps=1500, lr=0.1):
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    rng = np.random.default_rng(1000 + seed_run)
    Ws = init_weights(rng)
    W0 = [w.copy() for w in Ws]
    masks = init_masks()
    curve = train_loop(Xtr, ytr, Ws, masks, steps=steps, lr=lr,
                       seed=seed_run, X_val=Xv, y_val=yv, log_every=100)
    return {"test": evaluate(Xte, yte, Ws, masks),
            "Ws": Ws, "W0": W0, "masks": masks, "curve": curve}


def imp_search(seed_run, rounds=18, steps=1500, lr=0.1, rate=0.20):
    """Returns per-round (masks, W_trained, test_acc). Rewinds each round."""
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    rng = np.random.default_rng(1000 + seed_run)
    W0 = init_weights(rng)
    masks = init_masks()
    hist = []
    for r in range(rounds + 1):
        Ws = [w.copy() for w in W0]
        # apply current mask then train (round 0 = dense)
        Ws_m = [w.copy() for w in Ws]
        train_loop(Xtr, ytr, Ws_m, masks, steps=steps, lr=lr, seed=seed_run*100+r)
        acc = evaluate(Xte, yte, Ws_m, masks)
        hist.append({"round": r, "remaining": count_remaining(masks),
                     "test": acc, "masks": [m.copy() for m in masks],
                     "Wtrained": [w.copy() for w in Ws_m]})
        if r < rounds:
            masks = prune_layerwise(Ws_m, masks, rate=rate)
    return {"W0": W0, "hist": hist}


def train_with_mask(W0, masks, seed_run, steps=1500, lr=0.1, reinit=False,
                    sign_mode=None, data_seed=0):
    """reinit: fresh random weights. sign_mode: None | keep_sign_new_size |
    same_size | keep_size_new_sign | flip_all"""
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    if reinit:
        rng = np.random.default_rng(9000 + seed_run)
        from .model import init_weights as iw
        Ws = iw(rng)
    else:
        Ws = [w.copy() for w in W0]
    if sign_mode == "keep_sign_new_size":
        rng = np.random.default_rng(7000 + seed_run)
        Ws = [np.sign(w) * np.abs(rng.normal(0, np.sqrt(2.0/w.shape[0]), size=w.shape)) * m
              if i < 2 else np.sign(w) * np.abs(rng.normal(0, np.sqrt(2.0/w.shape[0]), size=w.shape)) * m
              for i, (w, m) in enumerate(zip(Ws, masks))]
        # keep exact sign of W0 where mask=1, fresh size
        Ws = [np.where(m > 0, np.sign(W0[i]) * np.abs(Ws[i]), 0.0) for i, m in enumerate(masks)]
    elif sign_mode == "same_size":
        Ws = [np.where(m > 0, np.sign(W0[i]) * np.mean(np.abs(W0[i][m > 0])), 0.0)
              for i, m in enumerate(masks)]
    elif sign_mode == "keep_size_new_sign":
        rng = np.random.default_rng(7001 + seed_run)
        Ws = [np.where(m > 0, np.abs(W0[i]) * np.sign(rng.normal(size=W0[i].shape)), 0.0)
              for i, m in enumerate(masks)]
    elif sign_mode == "flip_all":
        Ws = [-w.copy() * m for w, m in zip(Ws, masks)]
    else:
        Ws = [w * m for w, m in zip(Ws, masks)]
        # ensure pruned are exactly 0
        Ws = [np.where(m > 0, w, 0.0) for w, m in zip(Ws, masks)]
    # wrap as mutable list for training (train_loop mutates in place, masked)
    Ws = [w.copy() for w in Ws]
    curve = train_loop(Xtr, ytr, Ws, masks, steps=steps, lr=lr,
                       seed=seed_run, X_val=Xv, y_val=yv, log_every=100)
    return {"test": evaluate(Xte, yte, Ws, masks), "curve": curve, "Ws": Ws}
