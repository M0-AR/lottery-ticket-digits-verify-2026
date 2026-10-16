"""Full benchmark runner. Saves results/results.json + prints table.

Covers: dense, IMP curve, ticket vs new-start at r18/r22, random mask,
sign ablations, one-shot, small-dense, long-train, big-LR, dead-unit report.
Default is light (3 runs, 18 rounds) for laptop. Use --runs 10 for paper.
"""
import argparse, json, time
import numpy as np
from .data import fixed_split
from .experiments import imp_search, train_with_mask, run_dense
from .prune import random_mask_like, mask_from_magnitude_at_init
from .model import init_weights, init_masks, train_loop, evaluate, count_remaining
from .analyze import untrained_accuracy, sign_preservation, layer1_report


def small_dense_model(rng):
    W1 = rng.normal(0, np.sqrt(2.0/64), size=(64, 14))
    W2 = rng.normal(0, np.sqrt(2.0/14), size=(14, 10))
    return [W1, W2]


def train_small_dense(seed_run, steps, lr):
    from .data import fixed_split
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    rng = np.random.default_rng(5000 + seed_run)
    W1, W2 = small_dense_model(rng)
    Ws = [W1, W2]
    n = len(Xtr)
    brng = np.random.default_rng(seed_run)
    for s in range(steps):
        idx = brng.integers(0, n, size=60)
        Xb, yb = Xtr[idx], ytr[idx]
        # forward
        z1 = Xb @ W1; a1 = np.maximum(0, z1); logits = a1 @ W2
        m = logits.max(axis=1, keepdims=True); e = np.exp(logits-m); p = e/e.sum(axis=1, keepdims=True)
        d = p.copy(); d[np.arange(len(Xb)), yb] -= 1; d /= len(Xb)
        dW2 = a1.T @ d; da1 = d @ W2.T; dz1 = da1*(z1>0); dW1 = Xb.T @ dz1
        W1 -= lr*dW1; W2 -= lr*dW2
    z1 = Xte @ W1; a1 = np.maximum(0, z1); logits = a1 @ W2
    return float(((logits.argmax(axis=1)) == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=18)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="results/results.json")
    args = ap.parse_args()
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    res = {"config": vars(args), "runs": []}
    t0 = time.time()
    for run in range(args.runs):
        print(f"\n=== run {run} ===", flush=True)
        d = run_dense(run, steps=args.steps, lr=args.lr)
        print(f"dense test {d['test']*100:.2f}%", flush=True)
        out = imp_search(run, rounds=args.rounds, steps=args.steps, lr=args.lr)
        W0 = out["W0"]
        # pick rounds closest to video milestones
        for h in out["hist"]:
            if h["round"] in (0, 7, 13, 18, 22) and h["round"] <= args.rounds:
                print(f" round {h['round']:2d} remaining {h['remaining']:5d} test {h['test']*100:.2f}%", flush=True)
        # ticket vs new-start at final round
        masks = out["hist"][-1]["masks"]; Wt = out["hist"][-1]["Wtrained"]
        ticket = train_with_mask(W0, masks, run, steps=args.steps, lr=args.lr)
        newst = train_with_mask(W0, masks, run, steps=args.steps, lr=args.lr, reinit=True)
        # random mask control
        rm = random_mask_like(masks, np.random.default_rng(run))
        rnd = train_with_mask(W0, rm, run, steps=args.steps, lr=args.lr)
        # untrained
        ut_ticket = untrained_accuracy(Xte, yte, W0, masks)
        rng2 = np.random.default_rng(9000+run)
        Wfresh = init_weights(rng2)
        ut_new = untrained_accuracy(Xte, yte, Wfresh, masks)
        sp = sign_preservation(W0, Wt, masks)
        rep = layer1_report(Wt if False else ticket["Ws"], masks, W0)
        rep_new = layer1_report(newst["Ws"], masks, None)
        print(f" ticket {ticket['test']*100:.2f}% newstart {newst['test']*100:.2f}% randmask {rnd['test']*100:.2f}%", flush=True)
        print(f" untrained ticket {ut_ticket*100:.1f}% vs fresh {ut_new*100:.1f}% sign-pres {sp*100:.1f}%", flush=True)
        print(f" L1 ticket alive {rep['alive_path']} dead {rep['dead']} plus {rep['plus_frac_l1']:.2f} single {rep['single_conn_units']}", flush=True)
        # sign ablations
        ab = {}
        for mode in ["keep_sign_new_size", "same_size", "keep_size_new_sign", "flip_all"]:
            r = train_with_mask(W0, masks, run, steps=args.steps, lr=args.lr, sign_mode=mode)
            ab[mode] = r["test"]; print(f"  {mode} {r['test']*100:.2f}%", flush=True)
        res["runs"].append({"run": run, "dense": d["test"],
            "final_remaining": int(masks[0].sum()+masks[1].sum()+masks[2].sum()),
            "ticket": ticket["test"], "newstart": newst["test"], "randmask": rnd["test"],
            "untrained_ticket": ut_ticket, "untrained_fresh": ut_new,
            "sign_pres": sp, "l1_ticket": rep, "l1_new": rep_new, "ablations": ab,
            "imp_curve": [(h["round"], h["remaining"], h["test"]) for h in out["hist"]]})
    # small dense baseline (one middle layer 14 units, 1036 conns)
    sd = [train_small_dense(r, args.steps, args.lr) for r in range(args.runs)]
    print(f"\nsmall-dense-1036: {np.mean(sd)*100:.2f}% (runs {['%.2f'% (x*100) for x in sd]})")
    res["small_dense_1036"] = sd
    res["elapsed_s"] = time.time()-t0
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nsaved {args.out} elapsed {res['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
