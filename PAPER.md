# PAPER.md — Winning Tickets on Non-Negative Inputs: Signs, Death, and Masking-as-Training
*(Draft for journal submission; all numbers measured in this repo.)*

## Abstract
We replicate the lottery ticket hypothesis from an empty file on 1,797 8×8 handwritten digits with a 64–300–100–10 MLP (50,200 weights, no biases). Iterative magnitude pruning (20%/round, 10% output) with rewinding to initialization finds a 1,045-connection subnetwork (2.08%) at 96.0% test accuracy (dense 97.3%). The same mask with fresh initialization reaches 86.5%; random masks reach 18.3%. Sign ablations show the initialization's sign, not magnitude, carries the effect (same-sign/mean-size 96.3%, flipped-signs 20.9%). We identify the mechanism in this regime: non-negative inputs + ReLU + single-connection bottlenecks make negative signs absorbing dead states; IMP selects 74% positive signs (vs 50% at init), leaving ~11 dead units vs ~42 under reinit. The rewound ticket classifies 83.3% untrained (fresh 9.4%), showing pruning performed part of training. One-shot pruning loses 5–6 pp vs iterative; magnitude-at-init overlaps the ticket by ~20%. An ordinary 64–14–10 dense-small net (1,036 weights) reaches 96.2%, so the ticket is not a better small net here — it is a fragile sparse one whose learnability is sign-determined. We propose three one-flag falsifiers (zero-mean inputs, LeakyReLU, fan-in floor).

## 1 Introduction
Pruning compresses trained nets (Han et al. 2015) but sparse-from-scratch training fails. Frankle & Carbin (2019) showed rewinding survivors to `w0` resolves this on MNIST/CIFAR (Best Paper, ICLR). Zhou et al. (2019) decomposed the effect into zeros/signs/masking. We ask: what is load-bearing *in the minimal reproducible setting* — tiny images, shallow MLP, laptop budget — and what hidden variable explains "why can't the small net learn fresh?"

## 2 Related work
Covered in README §2 (9 sources, 2015–2020 + 2026 repro practice). Positions: (i) IMP>one-shot (Frankle); (ii) signs suffice, zeros matter, Supermask (Zhou); (iii) late-rewinding/stability for depth (Frankle 2019/2020); (iv) LR/schedule dispute (Liu vs Wang); (v) strong LTH existence proofs (Malach) and transfer (Morcos).

## 3 Method
### 3.1 Data
UCI digits via sklearn (1797×64, 0–16 → /16). Fixed stratified split 1000/297/500 (seed 0). Test N=500 → resolution 0.2 pp.
### 3.2 Model & optimization
He init, ReLU, softmax-CE, SGD bs=60, 1500 steps, lr 0.1 (sweep 0.05/0.1/0.2 flat). Masked updates. No biases (exact 50,200).
### 3.3 IMP with rewinding
Per-layer 20% (W3 10%), rewind to `w0`, up to 22 rounds. Controls: reinit, random mask, 4 sign modes, one-shot, mag-at-init, small-dense, 4× steps/LR.
### 3.4 Reproducibility
Docker, pinned deps, per-run RNG streams, JSON artifacts, val curves. 3 runs default (65 s), 10-run protocol supported.

## 4 Results
See README §5 table. Headlines (means, 3 runs): dense 97.33; r18 ticket 96.0; new-start 86.5; randmask 18.3; untrained 83.3 vs 9.4; sign-pres 99.7%; plus-frac 0.74; small-dense 96.2. All video directions reproduce; absolute Δ ≤1.5 pp except new-start gap larger here (stronger effect) and randmask lower (global-random harsher than layerwise-random — ablation noted).

## 5 Mechanism: sign-induced ReLU death
**Theorem (informal).** With X≥0 and ReLU, a hidden unit whose surviving incoming weights are all ≤0 is dead for all inputs and receives zero gradient (absorbing). Proof: z=Σwᵢxᵢ≤0 ∀x≥0 → a=0 → ∂L/∂w=0. Single-connection units die iff w<0.
Empirics: r18 L1 has 44–57 single-connection units of ~133 alive; ticket plus-frac 0.74 → ~11 dead; reinit coin-flip → ~42 dead. Capacity loss ≈ 23% of L1 before step 0. Flip-all → blind. Same-size-with-sign → full recovery. Predicts untrained-ticket elevation (mask kept large-|w| grown weights, 99.7% sign-preserved → already-separated representation).

## 6 Shortcuts & limits
Iterative beats one-shot by ~6 pp (better places). Mag-at-init ≈ chance overlap (ticket not readable from |w0|). Small-dense wins on wall-clock (dense kernels) — IMP's value here is scientific. Unstructured sparsity ≠ speedup on CPU (Frankle limits). 8×8 ≠ MNIST; N=500; shallow-only (no late-rewinding needed).

## 7 Proposed follow-ups (PhD-ready)
**E1 Zero-mean inputs** (X−0.5): predicts gap closes by >50%. **E2 LeakyReLU α=0.01**: predicts new-start rescue. **E3 Fan-in floor ≥4**: predicts gap erasure. Each is a 2-line change + rerun of `run_all.py`. Further: MNIST-64 vs MNIST-784 scaling; late-rewinding on 5-layer MLP; Supermask-only (no training) sweep; transfer across seeds/datasets (Morcos).

## 8 Conclusion
On non-negative-input shallow nets, the winning ticket is a list of places *plus signs*. IMP is a sign filter disguised as magnitude pruning. Small nets can learn alone when no unit hangs by a single negative thread.

## References
[1] Frankle & Carbin, LTH, ICLR 2019 (1803.03635). [2] Han et al., Deep Compression, NeurIPS 2015. [3] Zhou et al., Deconstructing LT, NeurIPS 2019 (1905.01067). [4] Frankle et al., Stabilizing LTH, 1903.01611. [5] Frankle et al., LMC & LTH, ICML 2020. [6] Liu et al., Rethinking Pruning, ICLR 2019. [7] Wang et al., Interpreting LT, ICLR 2020. [8] Ramanujan et al., What's Hidden..., NeurIPS 2019. [9] Malach et al., Proving LTH, ICML 2020. [10] Morcos et al., One Ticket..., 1906.02773. [11] Brix et al., Stabilized LTH for Transformers, 2005.03454. [12] Pedregosa et al., Scikit-learn, JMLR 2011 (digits data).
