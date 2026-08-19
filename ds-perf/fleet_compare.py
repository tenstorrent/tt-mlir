# Build DS vs no-DS per-matmul comparison tables across the model fleet.
#
# Reads the per-matmul CSVs produced by matmul_detail.py for each <model>__<variant>,
# groups by weight shape (K x N) -- which identifies the projection -- and reports
# time, achieved weight bandwidth, and the DS penalty per shape.
#
# Durations originate from percore_perf.py (per-core, epoch-safe), never from
# ops_perf_results.csv's DEVICE KERNEL DURATION.

import argparse
import csv
from collections import defaultdict
from pathlib import Path

BYTES_PER_ELEM = {"BFLOAT8_B": 1.0625, "BFLOAT4_B": 0.5625, "BFLOAT16": 2.0, "FLOAT32": 4.0}


def load(path):
    """weight shape (K,N) -> list of rows. matmul_detail.py already restricted
    its output to the traced region, so no further filtering here."""
    g = defaultdict(list)
    if not Path(path).exists():
        return None
    with open(path) as fh:
        for r in csv.DictReader(fh):
            g[(int(r["K_w"]), int(r["N"]))].append(r)
    return g


def stats(rows):
    ns = [float(r["ns"]) for r in rows]
    K, N = int(rows[0]["K_w"]), int(rows[0]["N"])
    mb = K * N * BYTES_PER_ELEM.get(rows[0].get("in1_dtype", "BFLOAT8_B"), 1.0625) / 1e6
    avg = sum(ns) / len(ns)
    return {
        "n": len(ns), "avg": avg, "total": sum(ns), "mb": mb,
        "gbs": mb * 1e3 / (avg / 1e3) if avg else 0,
        "cfg": rows[0].get("cfg", "?"), "cores": rows[0].get("cores", "?"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    summary = []
    for m in args.models:
        ds, nd = load(d / f"{m}__ds.matmuls.csv"), load(d / f"{m}__nods.matmuls.csv")
        print(f"\n{'='*104}\n{m}\n{'='*104}")
        if ds is None or nd is None:
            print(f"  missing data (ds={ds is not None}, nods={nd is not None})")
            continue
        print(f"{'K x N':>14s} {'n':>4s} | {'DS us':>9s} {'GB/s':>7s} {'cfg':>13s} | "
              f"{'noDS us':>9s} {'GB/s':>7s} {'cfg':>13s} | {'penalty':>8s} {'delta us':>9s}")
        print("-" * 104)
        tds = tnd = 0.0
        for key in sorted(set(ds) | set(nd), key=lambda k: -sum(float(r["ns"]) for r in ds.get(k, nd.get(k, [])))):
            a = stats(ds[key]) if key in ds else None
            b = stats(nd[key]) if key in nd else None
            K, N = key
            if a: tds += a["total"]
            if b: tnd += b["total"]
            if a and b:
                pen = a["avg"] / b["avg"] if b["avg"] else float("nan")
                flag = "  <<<" if pen >= 1.5 else ""
                print(f"{K:6d}x{N:<7d} {a['n']:4d} | {a['avg']/1e3:9.1f} {a['gbs']:7.1f} {a['cfg']:>13s} | "
                      f"{b['avg']/1e3:9.1f} {b['gbs']:7.1f} {b['cfg']:>13s} | {pen:7.2f}x "
                      f"{(a['total']-b['total'])/1e3:+9.1f}{flag}")
            elif a:
                print(f"{K:6d}x{N:<7d} {a['n']:4d} | {a['avg']/1e3:9.1f} {a['gbs']:7.1f} {a['cfg']:>13s} | "
                      f"{'--':>9s} {'--':>7s} {'--':>13s} | {'ds only':>8s}")
            else:
                print(f"{K:6d}x{N:<7d} {b['n']:4d} | {'--':>9s} {'--':>7s} {'--':>13s} | "
                      f"{b['avg']/1e3:9.1f} {b['gbs']:7.1f} {b['cfg']:>13s} | {'nods only':>8s}")
        print("-" * 104)
        ratio = tds / tnd if tnd else float("nan")
        print(f"{'matmul total':>14s} {'':4s} | {tds/1e3:9.1f} {'':7s} {'':13s} | {tnd/1e3:9.1f} "
              f"{'':7s} {'':13s} | {ratio:7.2f}x {(tds-tnd)/1e3:+9.1f}")
        summary.append((m, tds / 1e3, tnd / 1e3, ratio))

    print(f"\n{'='*72}\nFLEET SUMMARY -- matmul device time per decode step (traced region)\n{'='*72}")
    print(f"{'model':22s} {'DS us':>10s} {'noDS us':>10s} {'DS/noDS':>9s} {'verdict':>12s}")
    for m, a, b, r in summary:
        verdict = "DS WORSE" if r > 1.05 else ("DS better" if r < 0.95 else "parity")
        print(f"{m:22s} {a:10.1f} {b:10.1f} {r:8.2f}x {verdict:>12s}")


if __name__ == "__main__":
    main()
