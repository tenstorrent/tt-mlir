# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-side tracy profiling sweep for the softmax cases.

Runs `ttrt perf` on each flatbuffer with the device profiler enabled and
collects per-op DEVICE KERNEL DURATION [ns] from the generated
ops_perf_results.csv, emitting a single comparison table.

Why the flags:
  --fabric-config disabled : the flatbuffers are single-core / single-chip;
        fabric_1d (ttrt's default) reserves extra L1/DRAM and is irrelevant here.
  ttnn + d2m-fused are run with --ignore-version because their hand-authored /
        committed system_desc predates the profiler's L1 reservation. This is
        SAFE here: ttnn allocates L1 dynamically at runtime, and the d2m-fused
        source already carries l1_unreserved_base=103840 (== the profiler device),
        with the only diffs in DRAM/bank-mapping which a pure-L1 single-core run
        never touches.
  d2m-unfused MUST be generated under TT_METAL_DEVICE_PROFILER=1 (see gen note
        below) so its compile-time L1 addresses sit above the profiler buffer;
        those flatbuffers then match cleanly with no --ignore-version.

Generate the unfused flatbuffers first (one-time), under the profiler:
  TT_METAL_DEVICE_PROFILER=1 D2M_JIT_SAVE_FLATBUFFER_PATH=$PWD/_fb/d2m_unfused_NxN_prof.ttm \
      python3 probe.py softmax N

Usage:
  python3 devperf.py            # full sweep over _fb (override with FB_DIR)
"""
import csv
import json
import os
import subprocess
import sys

FB = os.environ.get("FB_DIR", "_fb")
ARTIFACT = os.path.join(os.getcwd(), "perf_artifacts")
REPORT = (
    "build/python_packages/ttrt/runtime/generated/profiler/reports/ops_perf_results.csv"
)

# (label, flatbuffer, needs_ignore_version)
CASES = [
    ("ttnn 1x1", "ttnn_softmax_1x1.ttnn", True),
    ("ttnn 2x2", "ttnn_softmax_2x2.ttnn", True),
    ("ttnn 3x3", "ttnn_softmax_3x3.ttnn", True),
    ("d2m-fused 1x1", "d2m_fused_1x1.ttm", True),
    ("d2m-fused 2x2", "d2m_fused_2x2.ttm", True),
    ("d2m-fused 3x3", "d2m_fused_3x3.ttm", True),
    ("d2m-unfused 1x1", "d2m_unfused_1x1_prof.ttm", False),
    ("d2m-unfused 2x2", "d2m_unfused_2x2_prof.ttm", False),
    ("d2m-unfused 3x3", "d2m_unfused_3x3_prof.ttm", False),
]


def short_opcode(s):
    s = s or ""
    if s.startswith('loc("'):
        # d2m generic with file location: keep the file:line:col tail.
        tail = s.rstrip(')"').split('"')[-1].lstrip(":")
        return "generic@" + tail
    if s.startswith("loc("):
        # d2m generic without resolved location (loc(unknown)).
        return "generic(unknown)"
    return s


def run_case(label, fb, ignore):
    path = os.path.join(FB, fb)
    if not os.path.exists(path):
        print(f"SKIP {label}: missing {path}")
        return None
    cmd = [
        "ttrt",
        "perf",
        path,
        "--fabric-config",
        "disabled",
        "--artifact-dir",
        ARTIFACT,
    ]
    if ignore:
        cmd.append("--ignore-version")
    # timeout -s KILL so a hung run never orphans + holds the chip lock.
    full = ["timeout", "-s", "KILL", "300"] + cmd
    subprocess.run(full, capture_output=True, text=True)
    try:
        res = json.load(open("perf_results.json"))[0].get("result")
    except Exception:
        res = "no-json"
    if res != "pass":
        print(f"FAIL {label}: result={res}")
        return None
    ops = []
    with open(REPORT) as f:
        for r in csv.DictReader(f):
            ops.append(
                (
                    short_opcode(r.get("OP CODE")),
                    int(r.get("DEVICE KERNEL DURATION [ns]") or 0),
                )
            )
    return ops


def main():
    results = {}
    for label, fb, ignore in CASES:
        ops = run_case(label, fb, ignore)
        if ops is None:
            continue
        total = sum(d for _, d in ops)
        results[label] = (ops, total)
        print(f"\n=== {label}: {len(ops)} device op(s), total kernel = {total} ns ===")
        for name, d in ops:
            print(f"    {name:42} {d:>8} ns")
    print("\n\n================ SUMMARY (total device kernel ns) ================")
    print(f"{'case':18} {'1x1':>10} {'2x2':>10} {'3x3':>10}")
    for fam in ("ttnn", "d2m-fused", "d2m-unfused"):
        row = [fam]
        for sz in ("1x1", "2x2", "3x3"):
            key = f"{fam} {sz}"
            row.append(str(results[key][1]) if key in results else "-")
        print(f"{row[0]:18} {row[1]:>10} {row[2]:>10} {row[3]:>10}")


if __name__ == "__main__":
    main()
