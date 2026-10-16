"""Analysis helpers: dead units, sign preservation, untrained accuracy."""
import numpy as np
from .model import forward, accuracy, dead_units_layer1


def untrained_accuracy(X, y, Ws, masks):
    logits, _ = forward(X, Ws, masks)
    return accuracy(logits, y)


def sign_preservation(W0, Wtrained, masks):
    """Fraction of surviving conns whose final sign == initial sign."""
    tot = 0; same = 0
    for w0, wt, m in zip(W0, Wtrained, masks):
        sel = m > 0
        s0 = np.sign(w0[sel]); s1 = np.sign(wt[sel])
        tot += sel.sum(); same += (s0 == s1).sum()
    return float(same / max(1, tot))


def layer1_report(Ws, masks, W0=None):
    alive, dead = dead_units_layer1(Ws, masks)
    # plus-fraction among surviving layer-1 inits
    if W0 is not None:
        sel = masks[0] > 0
        plus = float((W0[0][sel] > 0).mean())
    else:
        plus = None
    # connection count detail
    import numpy as np
    per_unit = masks[0].sum(axis=0)  # 300
    alive_units = int(((masks[0].sum(axis=0) > 0) & (masks[1].sum(axis=1) > 0)).sum())
    single = int((((masks[0].sum(axis=0) == 1) & (masks[1].sum(axis=1) > 0))).sum())
    return {"alive_path": alive, "dead": dead, "plus_frac_l1": plus,
            "alive_units": alive_units, "single_conn_units": single,
            "l1_conns": int(masks[0].sum())}
