"""Benchmark table generator from results/results.json."""
import json
with open("results/results.json") as f:
    r = json.load(f)
print(f"runs={len(r['runs'])} elapsed={r['elapsed_s']:.1f}s")
for x in r["runs"]:
    print(x["run"], f"dense={x['dense']*100:.2f}", f"ticket={x['ticket']*100:.2f}",
          f"new={x['newstart']*100:.2f}", f"rand={x['randmask']*100:.2f}",
          f"rem={x['final_remaining']}")
print("small_dense", [f"{v*100:.2f}" for v in r["small_dense_1036"]])
