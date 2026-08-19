# Per-matmul breakdown for the traced decode step.
#
# Durations come from percore_perf.py's output (valid), metadata from
# ops_perf_results.csv (its shapes/attributes are fine; only its durations are
# corrupted by the cross-core clock epoch bug). Joined on GLOBAL CALL COUNT.

import argparse
import csv
import re
from collections import defaultdict

import pandas as pd


def program_config(attrs):
    """Which matmul program config ttnn picked, from the ATTRIBUTES blob."""
    if not isinstance(attrs, str):
        return "?"
    # Runtime spells these MatmulMultiCoreReuse...DRAMSharded / ...MultiCast1D...
    if "DRAMSharded" in attrs:
        return "DRAM-sharded"
    if "MultiCast1D" in attrs:
        return "1D mcast"
    if "MultiCast" in attrs:
        return "2D mcast"
    if "MultiCoreReuse" in attrs:
        return "reuse"
    return "default"


def gather_in0(attrs):
    if isinstance(attrs, str):
        m = re.search(r"gather_in0'?\s*[:=]\s*'?(\w+)", attrs)
        if m:
            return m.group(1)
    return ""


def dim(v):
    """Shape cells are written 'padded[logical]'; the padded extent is what runs."""
    s = str(v)
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else 0


def short_mem(v):
    if not isinstance(v, str):
        return "?"
    return (v.replace("DEV_0_", "").replace("DEV_1_", "")
             .replace("INTERLEAVED", "IL").replace("WIDTH_SHARDED", "WS")
             .replace("HEIGHT_SHARDED", "HS").replace("BLOCK_SHARDED", "BS"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percore", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dur, cores_of = {}, {}
    with open(args.percore) as fh:
        for r in csv.DictReader(fh):
            if r["op_name"] != "MatmulDeviceOperation":
                continue
            if not r["replay"].strip():
                continue  # traced region only
            dur[int(r["op_id"])] = float(r["duration_ns"])
            cores_of[int(r["op_id"])] = int(r["cores"])

    df = pd.read_csv(args.report, low_memory=False)
    mm = df[df["OP CODE"] == "MatmulDeviceOperation"].copy()
    mm["call"] = pd.to_numeric(mm["GLOBAL CALL COUNT"], errors="coerce")

    rows, seen = [], set()
    for _, r in mm.iterrows():
        call = int(r["call"])
        if call not in dur or call in seen:
            continue
        seen.add(call)
        M = dim(r["INPUT_0_Y_PAD[LOGICAL]"]); K = dim(r["INPUT_0_X_PAD[LOGICAL]"])
        K1 = dim(r["INPUT_1_Y_PAD[LOGICAL]"]); N = dim(r["INPUT_1_X_PAD[LOGICAL]"])
        ideal = pd.to_numeric(r.get("PM IDEAL [ns]"), errors="coerce")
        fpu = pd.to_numeric(r.get("PM FPU UTIL (%)"), errors="coerce")
        d = dur[call]
        rows.append({
            "call": call, "M": M, "K": K, "N": N, "K_w": K1,
            "cores": cores_of[call],
            "ns": d,
            "cfg": program_config(r.get("ATTRIBUTES")),
            "gather_in0": gather_in0(r.get("ATTRIBUTES")),
            "in0_mem": short_mem(r.get("INPUT_0_MEMORY")),
            "in1_mem": short_mem(r.get("INPUT_1_MEMORY")),
            "in1_dtype": str(r.get("INPUT_1_DATATYPE", "")).replace("DataType::", ""),
            "out_mem": short_mem(r.get("OUTPUT_0_MEMORY")),
            "bias": "y" if pd.notna(r.get("INPUT_2_Y_PAD[LOGICAL]")) else "",
            "fidelity": str(r.get("MATH FIDELITY", "")),
            "pm_ideal_ns": float(ideal) if pd.notna(ideal) else None,
            "fpu_util": float(fpu) if pd.notna(fpu) else None,
        })

    rows.sort(key=lambda r: r["call"])
    total = sum(r["ns"] for r in rows)
    print(f"traced matmuls matched: {len(rows)}   total {total/1e3:.1f} us\n")

    # Group by weight shape: that identifies the projection.
    groups = defaultdict(list)
    for r in rows:
        groups[(r["K_w"], r["N"], r["M"])].append(r)

    print("=== by weight shape (K x N), batch rows M ===")
    hdr = (f"{'K x N':>14s} {'M':>5s} {'n':>4s} {'cfg':>13s} {'in1':>10s} {'dtype':>8s} "
           f"{'cores':>6s} {'avg us':>8s} {'min us':>8s} {'max us':>8s} {'total us':>9s} {'%':>6s} {'ideal us':>9s} {'eff%':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups, key=lambda k: -sum(r["ns"] for r in groups[k])):
        g = groups[key]
        K_w, N, M = key
        ns = [r["ns"] for r in g]
        tot = sum(ns)
        ideals = [r["pm_ideal_ns"] for r in g if r["pm_ideal_ns"]]
        ideal_avg = sum(ideals) / len(ideals) if ideals else None
        eff = (100 * ideal_avg / (tot / len(g))) if ideal_avg else None
        print(f"{K_w:6d}x{N:<7d} {M:5d} {len(g):4d} {g[0]['cfg']:>13s} {g[0]['in1_mem']:>10s} "
              f"{g[0]['in1_dtype']:>8s} {g[0]['cores']:6d} {tot/len(g)/1e3:8.1f} {min(ns)/1e3:8.1f} "
              f"{max(ns)/1e3:8.1f} {tot/1e3:9.1f} {100*tot/total:5.1f}% "
              f"{ideal_avg/1e3 if ideal_avg else float('nan'):9.1f} {eff if eff else float('nan'):6.1f}")

    print(f"\n=== every traced matmul, slowest first ===")
    hdr2 = (f"{'call':>9s} {'M':>5s} {'K':>6s} {'N':>7s} {'cores':>6s} {'us':>8s} "
            f"{'cfg':>13s} {'in0':>10s} {'in1':>10s} {'out':>10s} {'bias':>4s} {'ideal us':>9s} {'fpu%':>6s}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in sorted(rows, key=lambda r: -r["ns"]):
        print(f"{r['call']:9d} {r['M']:5d} {r['K']:6d} {r['N']:7d} {r['cores']:6d} {r['ns']/1e3:8.1f} "
              f"{r['cfg']:>13s} {r['in0_mem']:>10s} {r['in1_mem']:>10s} {r['out_mem']:>10s} {r['bias']:>4s} "
              f"{r['pm_ideal_ns']/1e3 if r['pm_ideal_ns'] else float('nan'):9.1f} "
              f"{r['fpu_util'] if r['fpu_util'] else float('nan'):6.1f}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: -r["ns"]))
    print(f"\nper-matmul CSV -> {args.out}")


if __name__ == "__main__":
    main()
