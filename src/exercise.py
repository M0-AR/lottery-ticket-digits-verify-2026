"""Exercise: IMP with 3 rounds (video's closing question).

Guess its score first, then run:
  python src/exercise.py --rounds 3 --runs 3
Expected: 3 rounds of 20% pruning -> 0.8^3 = 51.2% left (~25702 conns).
Ticket should roughly match/slightly beat dense (~98%), because early
rounds often improve accuracy (regularization). New-start should be close
at this density (gap opens only at high sparsity).
"""
import argparse, json
import numpy as np
from .experiments import imp_search, train_with_mask
from .data import fixed_split
from .model import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()
    (Xtr, ytr), (Xv, yv), (Xte, yte) = fixed_split()
    print(f"EXERCISE: {args.rounds} rounds x {args.runs} runs, {args.steps} steps, lr={args.lr}")
    print(f"QUESTION: guess the ticket score before running. Answer below after run.")
    for run in range(args.runs):
        out = imp_search(run, rounds=args.rounds, steps=args.steps, lr=args.lr)
        W0 = out["W0"]; masks = out["hist"][-1]["masks"]
        rem = out["hist"][-1]["remaining"]
        t = train_with_mask(W0, masks, run, steps=args.steps, lr=args.lr)
        n = train_with_mask(W0, masks, run, steps=args.steps, lr=args.lr, reinit=True)
        print(f"run {run}: remaining={rem} ticket={t['test']*100:.2f}% newstart={n['test']*100:.2f}%")
    print("ANSWER: with 3 rounds (~51% left) both ticket and new-start stay near dense (~97-98%).")
    print("The ticket advantage only opens at high sparsity (18-22 rounds).")


if __name__ == "__main__":
    main()
