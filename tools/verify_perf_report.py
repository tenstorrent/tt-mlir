#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-check tt-mlir's ttnn-collect-perf-metrics "flops" report.

This validates the report against INDEPENDENT ground truth rather than trusting
it, so bugs like the earlier peak contamination get caught:

  1. Peak re-derivation: recompute peak_flops_per_sec from first principles
     (cores * clock * 2*32^3 / cycles_per_tile) and compare to the report. This
     is a separate code path from the compiler, so it catches formula/constant
     drift.
  2. Published-peak anchor: print tt-metal's documented full-chip peak so a
     wildly-off number is obvious to the eye.
  3. Internal invariant: sum(per_op[].flops) == total_flops (needs a report
     produced with ttnn-perf-metrics-verbose-output-enabled=true).
  4. Physical bound: with a measured step time, exact MFU must be <= 100%.
     Exceeding it means the peak is wrong or the graph ran a faster precision
     than the reference.

Usage:
  python tools/verify_perf_report.py <report_*.json> [<more.json> ...]
      [--step-time-s 0.0123] [--fidelity hifi4]
      [--cores 64] [--clock-ghz 1.0]   # override if no perf_targets block

For the strongest check (device-measured), separately run the OpModel device
test (TestOpModelInterface --gtest_filter=*MatmulRuntimeVsPeak*): it measures a
matmul's runtime on real silicon via getOpRuntime and prints achieved TFLOP/s to
compare against the report's peak.
"""
import argparse
import glob
import json
import sys

FLOPS_PER_TILE_MUL = 2 * 32 * 32 * 32  # 65536
CYCLES_PER_TILE = {"lofi": 16, "hifi2": 32, "hifi3": 48, "hifi4": 64}

# Documented per-arch defaults (tt-metal tech_reports/GEMM_FLOPS), used only when
# the report has no perf_targets block to read cores/clock from.
ARCH_DEFAULTS = {
    "wormhole_b0": {"cores": 64, "clock_ghz": 1.0, "observed_peak_tflops": 190.0},
    "blackhole": {"cores": 130, "clock_ghz": 1.35, "observed_peak_tflops": 580.0},
}


def peak_flops(cores, clock_hz, fidelity):
    return cores * clock_hz * FLOPS_PER_TILE_MUL / CYCLES_PER_TILE[fidelity]


class Checker:
    def __init__(self):
        self.failures = 0

    def check(self, ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            self.failures += 1


def verify_report(path, args, chk):
    print(f"\n=== {path} ===")
    with open(path) as f:
        data = json.load(f)
    flops = data.get("flops")
    if not isinstance(flops, dict):
        print("  (no 'flops' section - stale tt-mlir plugin? skipping)")
        return
    perf = data.get("perf_targets", {})

    # Resolve cores/clock: prefer the report's own hardware block, else defaults.
    cores = args.cores
    clock_hz = args.clock_ghz * 1e9 if args.clock_ghz else None
    arch = perf.get("arch")
    if cores is None and perf.get("num_tensix_cores"):
        cores = perf["num_tensix_cores"]
    if clock_hz is None and perf.get("aiclk_hz"):
        clock_hz = perf["aiclk_hz"]
    if (cores is None or clock_hz is None) and arch in ARCH_DEFAULTS:
        cores = cores or ARCH_DEFAULTS[arch]["cores"]
        clock_hz = clock_hz or ARCH_DEFAULTS[arch]["clock_ghz"] * 1e9
    if cores is None or clock_hz is None:
        print("  (no cores/clock available - pass --cores/--clock-ghz)")
        return
    print(
        f"  arch={arch} cores={cores} clock={clock_hz / 1e9:.3f}GHz "
        f"chips_used={flops.get('num_chips_used')}"
    )

    # 1. Peak re-derivation (independent of the compiler's computation).
    report_peak = flops.get("peak_flops_per_sec", {})
    for fid in CYCLES_PER_TILE:
        if fid not in report_peak:
            continue
        expected = peak_flops(cores, clock_hz, fid)
        got = report_peak[fid]
        rel = abs(got - expected) / expected if expected else 1.0
        chk.check(
            rel < 1e-6,
            f"peak[{fid}] report={got / 1e12:.2f} TFLOP/s vs re-derived "
            f"{expected / 1e12:.2f} TFLOP/s",
        )

    # 2. Published-peak anchor (eyeball only).
    if arch in ARCH_DEFAULTS:
        lofi = peak_flops(cores, clock_hz, "lofi") / 1e12
        obs = ARCH_DEFAULTS[arch]["observed_peak_tflops"]
        print(
            f"  [note] theoretical LoFi peak {lofi:.0f} TFLOP/s; tt-metal "
            f"observed full-chip peak ~{obs:.0f} TFLOP/s (theoretical > observed)"
        )

    # 3. sum(per_op) == total_flops (verbose reports only).
    per_op = flops.get("per_op")
    if per_op:
        s = sum(op.get("flops", 0) for op in per_op)
        chk.check(
            s == flops.get("total_flops"),
            f"sum(per_op flops)={s} == total_flops={flops.get('total_flops')}",
        )
    else:
        print("  (no per_op array - rerun with verbose to check the FLOP sum)")

    # 4. MFU <= 100% physical bound.
    if args.step_time_s:
        fid = args.fidelity
        peak = report_peak.get(fid) or peak_flops(cores, clock_hz, fid)
        chips = max(1, flops.get("num_chips_used", 1))
        total = flops.get("total_flops", 0)
        # per-chip counted flops / (step * per-chip peak)
        mfu = total / (args.step_time_s * peak) * 100.0 if peak else 0.0
        achieved = total * chips / args.step_time_s / 1e12
        print(
            f"  measured step={args.step_time_s * 1e3:.2f} ms -> exact MFU "
            f"{mfu:.2f}% ({achieved:.2f} TFLOP/s achieved, vs {fid})"
        )
        chk.check(
            mfu <= 100.0 + 1e-6,
            f"exact MFU {mfu:.2f}% <= 100% (else wrong peak/fidelity)",
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reports", nargs="+", help="perf-metrics JSON file(s) or globs")
    ap.add_argument(
        "--step-time-s",
        type=float,
        default=None,
        help="measured step time (s) to check MFU <= 100%%",
    )
    ap.add_argument(
        "--fidelity",
        default="hifi4",
        choices=list(CYCLES_PER_TILE),
        help="reference math fidelity for the MFU check",
    )
    ap.add_argument(
        "--cores", type=int, default=None, help="override Tensix core count (per chip)"
    )
    ap.add_argument(
        "--clock-ghz", type=float, default=None, help="override AICLK in GHz"
    )
    args = ap.parse_args()

    paths = [p for pat in args.reports for p in sorted(glob.glob(pat)) or [pat]]
    chk = Checker()
    for p in paths:
        try:
            verify_report(p, args, chk)
        except (OSError, json.JSONDecodeError) as e:
            print(f"\n=== {p} ===\n  (could not read: {e})")
            chk.failures += 1

    print(f"\n{'FAILED' if chk.failures else 'OK'}: {chk.failures} check(s) failed")
    sys.exit(1 if chk.failures else 0)


if __name__ == "__main__":
    main()
