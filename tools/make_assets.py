"""Generate all README/docs assets from measured results/results.json + live data.
Verified: run `python tools/make_assets.py` -> PNGs + GIF in docs/.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from sklearn.datasets import load_digits

os.makedirs("docs", exist_ok=True)
with open("results/results.json") as f:
    R = json.load(f)

runs = R["runs"]
# 1. digit samples (real data)
d = load_digits()
fig, ax = plt.subplots(2, 8, figsize=(10, 3))
for i in range(16):
    a = ax[i//8][i%8]
    a.imshow(d.images[i], cmap="gray", interpolation="nearest")
    a.set_title(f"{d.target[i]}", fontsize=10)
    a.axis("off")
fig.suptitle("Real public data: 1797 handwritten digits, 8x8 (values 0-16 -> scaled 0-1)")
fig.tight_layout()
fig.savefig("docs/digits.png", dpi=150)
plt.close(fig)

# 2. accuracy vs sparsity (IMP curve, mean of runs)
rounds = sorted(set(r for run in runs for (r, _, _) in run["imp_curve"]))
rem_mean, acc_mean = [], []
for r in rounds:
    vals = [(rem, acc) for run in runs for (rr, rem, acc) in run["imp_curve"] if rr == r]
    rem_mean.append(np.mean([v[0] for v in vals]))
    acc_mean.append(np.mean([v[1] for v in vals])*100)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(rem_mean, acc_mean, marker="o")
ax.set_xscale("log")
ax.set_xlabel("Connections remaining (log scale)")
ax.set_ylabel("Test accuracy % (500 held-out)")
ax.set_title("IMP: accuracy holds while 98% of connections are deleted")
ax.grid(True, which="both", alpha=0.3)
for x, y, r in zip(rem_mean, acc_mean, rounds):
    if r in (0, 7, 13, 18):
        ax.annotate(f"r{r}\n{x:.0f}\n{y:.1f}%", (x, y), textcoords="offset points", xytext=(8, -12), fontsize=8)
fig.tight_layout()
fig.savefig("docs/accuracy_vs_sparsity.png", dpi=150)
plt.close(fig)

# 3. ticket vs new-start vs random
labels = [f"run{x['run']}" for x in runs]
t = [x["ticket"]*100 for x in runs]
n = [x["newstart"]*100 for x in runs]
rm = [x["randmask"]*100 for x in runs]
x = np.arange(len(runs))
fig, ax = plt.subplots(figsize=(7, 4))
w = 0.25
ax.bar(x-w, t, w, label="Ticket (mask+w0)")
ax.bar(x, n, w, label="New-start (same mask, fresh numbers)")
ax.bar(x+w, rm, w, label="Random mask")
ax.set_xticks(x, labels)
ax.set_ylabel("Test % @1045 conns")
ax.set_title("Same places, different numbers: initialization decides")
ax.legend(fontsize=8)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("docs/ticket_vs_new.png", dpi=150)
plt.close(fig)

# 4. sign ablations (mean)
modes = ["keep_sign_new_size", "same_size", "keep_size_new_sign", "flip_all"]
names = ["keep sign\nnew size", "same size\n(all mean)", "keep size\nnew sign", "flip all\nsigns"]
vals = [np.mean([run["ablations"][m] for run in runs])*100 for m in modes]
base_ticket = np.mean(t)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(names, vals)
ax.axhline(base_ticket, color="green", linestyle="--", label=f"ticket {base_ticket:.1f}%")
ax.axhline(np.mean(n), color="red", linestyle="--", label=f"new-start {np.mean(n):.1f}%")
ax.set_ylabel("Test %")
ax.set_title("Signs are the ticket; sizes hardly matter")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("docs/signs.png", dpi=150)
plt.close(fig)

# 5. dead units
dt = [x["l1_ticket"]["dead"] for x in runs]
dn = [x["l1_new"]["dead"] for x in runs]
alive = [x["l1_ticket"]["alive_path"] for x in runs]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(runs))
ax.bar(x-0.2, dt, 0.4, label="Ticket dead units")
ax.bar(x+0.2, dn, 0.4, label="New-start dead units")
ax.set_xticks(x, labels)
ax.set_ylabel("Dead L1 units (of ~133 alive-path)")
ax.set_title("Fresh random signs kill 1/3 of layer 1 before training")
ax.legend()
fig.tight_layout()
fig.savefig("docs/dead_units.png", dpi=150)
plt.close(fig)

# 6. demo GIF: pruning sweep animation (remaining shrinks, acc holds)
fig, ax = plt.subplots(figsize=(6, 3.8))
ims = []
# build frames: progressive reveal of curve + shrinking bar
frames = list(zip(rem_mean, acc_mean, rounds))
def draw_frame(k):
    ax.clear()
    xs, ys = rem_mean[:k+1], acc_mean[:k+1]
    ax.plot(rem_mean, acc_mean, color="lightgray")
    ax.plot(xs, ys, marker="o", color="#2563eb")
    ax.set_xscale("log")
    ax.set_xlim(min(rem_mean)*0.8, max(rem_mean)*1.2)
    ax.set_ylim(min(acc_mean)-3, 100)
    ax.set_xlabel("Connections remaining")
    ax.set_ylabel("Test %")
    r = rounds[k]
    ax.set_title(f"Round {r}: {rem_mean[k]:.0f} conns -> {acc_mean[k]:.1f}%  (delete smallest 20%)")
    ax.grid(True, alpha=0.3)
fig2, ax2 = plt.subplots(figsize=(6, 3.8))
writer = PillowWriter(fps=2)
with writer.saving(fig2, "docs/demo.gif", dpi=100):
    for k in range(len(frames)):
        ax2.clear()
        ax2.plot(rem_mean, acc_mean, color="lightgray")
        ax2.plot(rem_mean[:k+1], acc_mean[:k+1], marker="o", color="#2563eb")
        ax2.set_xscale("log")
        ax2.set_xlim(min(rem_mean)*0.8, max(rem_mean)*1.2)
        ax2.set_ylim(min(acc_mean)-3, 100)
        ax2.set_xlabel("Connections remaining (log)")
        ax2.set_ylabel("Test accuracy %")
        ax2.set_title(f"Round {rounds[k]}: {rem_mean[k]:.0f} left -> {acc_mean[k]:.1f}%")
        ax2.grid(True, alpha=0.3)
        writer.grab_frame()
plt.close("all")
print("assets written to docs/:", sorted(os.listdir("docs")))
