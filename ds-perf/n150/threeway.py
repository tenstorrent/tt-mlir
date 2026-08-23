# Three-way end-to-end comparison: no-DS vs DS vs DS+collapse-guard.
#
#   before  = tt-mlir 99c621da (merge base, DS off)
#   after   = tt-mlir ae602833 (merge base + the 18-commit DS series)
#   patched = tt-mlir 1fad065b (= ae602833 + the collapse guard, one commit)
#
# The guard's effect is therefore isolated to the after -> patched column.
import csv
import glob
import json
import statistics as st
from pathlib import Path

VARIANTS = ["before", "after", "patched"]
LABEL = {"before": "no DS", "after": "DS", "patched": "DS+guard"}


def rep(variant, model):
    f = glob.glob(f"raw/{variant}/{model}/perf_report_*.json")
    if not f:
        return None
    r = json.load(open(f[0]))
    d = {x["measurement_name"]: x["value"] for x in r.get("measurements", [])}
    if not d.get("total_time"):
        return None
    bs = r.get("batch_size") or 1
    s = d["total_samples"]
    tokens = s / bs if s / bs >= 16 else s
    if tokens <= 1:
        return None
    return {
        "step_ms": 1000 * (d["total_time"] - (d.get("ttft") or 0) / 1000) / (tokens - 1),
        "ttft": d.get("ttft"), "layers": r.get("num_layers"),
    }


def main():
    root = Path("raw")
    keys = sorted({p.name for v in VARIANTS for p in (root / v).iterdir() if p.is_dir()})
    guard = list(csv.DictReader(open("roles_g1_guard.csv")))
    declined = {}
    for r in guard:
        if r["kind_before"] == "DS" and r["kind_after"] != "DS":
            declined.setdefault(r["model"], []).append(f"{r['role']}")

    rows = []
    for k in keys:
        m = {v: rep(v, k) for v in VARIANTS}
        if not all(m.values()):
            continue
        rows.append({"model": k, **{v: m[v]["step_ms"] for v in VARIANTS},
                     "ttft": {v: m[v]["ttft"] for v in VARIANTS},
                     "declined": declined.get(k, [])})

    print("## Decode step, three ways (ms)\n")
    print("| model | no DS | DS | DS+guard | DS vs no-DS | guard vs DS | guard vs no-DS | shapes the guard declined |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (not r["declined"], r["model"])):
        b, a, p = r["before"], r["after"], r["patched"]
        print(f"| {r['model']} | {b:.2f} | {a:.2f} | {p:.2f} "
              f"| {100*(a-b)/b:+.2f}% | {100*(p-a)/a:+.2f}% | {100*(p-b)/b:+.2f}% "
              f"| {', '.join(r['declined']) if r['declined'] else '—'} |")

    hit = [r for r in rows if r["declined"]]
    miss = [r for r in rows if not r["declined"]]
    for name, sel in (("models the guard changed", hit), ("models it did not touch", miss)):
        if not sel:
            continue
        g = [100 * (r["patched"] - r["after"]) / r["after"] for r in sel]
        d = [100 * (r["after"] - r["before"]) / r["before"] for r in sel]
        print(f"\n**{name}** (n={len(sel)}): DS vs no-DS median {st.median(d):+.2f}%, "
              f"guard vs DS median {st.median(g):+.2f}% "
              f"(range {min(g):+.2f}%..{max(g):+.2f}%)")


if __name__ == "__main__":
    main()
