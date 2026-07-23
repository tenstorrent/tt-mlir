#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Summarize a chisel JSONL report: rank problematic ops by PCC / status."""
import argparse
from collections import defaultdict

from chisel.report import ChiselReport, RecordStatus, PASS_STATUS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--pcc-threshold", type=float, default=0.99)
    args = ap.parse_args()

    report = ChiselReport.from_jsonl(args.jsonl)
    records = report.records
    print(f"total records: {len(records)}")

    # Status histogram
    by_status = defaultdict(int)
    for r in records:
        by_status[r.status.value] += 1
    print("\n=== status histogram ===")
    for k, v in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"  {v:6d}  {k}")

    # Numerics records: worst PCC per op per mode
    numerics = [r for r in records if r.check == "numerics"
                and hasattr(r.payload, "pcc")]
    print(f"\n=== numerics records: {len(numerics)} ===")

    for mode in ("isolated", "accumulated"):
        rows = [r for r in numerics if getattr(r.payload, "mode", None)
                and r.payload.mode.value == mode]
        if not rows:
            continue
        # worst pcc per op
        worst = {}
        for r in rows:
            p = r.payload.pcc
            if r.op not in worst or p < worst[r.op][0]:
                worst[r.op] = (p, r.ssa)
        print(f"\n--- {mode}: worst PCC per op (asc) ---")
        for op, (pcc, ssa) in sorted(worst.items(), key=lambda x: x[1][0]):
            flag = "  <-- BELOW" if pcc < args.pcc_threshold else ""
            print(f"  {pcc:8.5f}  {op:45s} {ssa or ''}{flag}")

    # Failures (non-pass status)
    fails = [r for r in records if r.status not in PASS_STATUS]
    print(f"\n=== failing records (non-pass status): {len(fails)} ===")
    seen = set()
    for r in fails:
        key = (r.op, r.status.value)
        if key in seen:
            continue
        seen.add(key)
        extra = ""
        if hasattr(r.payload, "pcc"):
            extra = f" pcc={r.payload.pcc:.5f}"
        if hasattr(r.payload, "traceback"):
            extra = " " + r.payload.traceback.splitlines()[-1][:120]
        print(f"  {r.status.value:20s} {r.op:40s}{extra}")


if __name__ == "__main__":
    main()
