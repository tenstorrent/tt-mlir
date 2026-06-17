# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Virtual-grid scaling sweep for d2m softmax.

For each (N_tile_rows, NC_tile_cols, G_grid_rows) config, runs probe.py under the
device profiler (d2m_jit's own runtime — ttrt can't load virtual-grid flatbuffers),
then parses profile_log_device.csv for the per-core softmax compute time (the
longest TRISC-KERNEL interval on each core) and the number of physical cores used.

This sidesteps two things: (1) ttrt's variant error on virtual-grid .ttm, and
(2) the [M,N] kernel-compile hang for M>2 / wide reductions (we use square blocks
[n,n] -> n*n tiles/core, which compile).

Graphs:
  - 1 tile/core: 1x1 (1 core) vs 1x64 (64 cores)  -> parallelism keeps per-core flat
  - tiles/core scaling on 64 cores: 1, 4, 9        -> per-core compute vs work
"""
import csv
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
LOG = os.path.join(
    ROOT,
    "third_party/tt-metal/src/tt-metal/generated/profiler/.logs/profile_log_device.csv",
)


def per_core_kernel_ns():
    """From profile_log_device.csv: per-core longest TRISC-KERNEL interval (= the
    softmax compute generic) and #cores. 1 cycle == 1 ns at 1 GHz."""
    if not os.path.exists(LOG):
        return None, 0
    ev = defaultdict(list)
    with open(LOG) as f:
        next(f)  # ARCH header
        for r in csv.DictReader(f):
            if r[" zone name"].strip() != "TRISC-KERNEL":
                continue
            core = (r[" core_x"].strip(), r[" core_y"].strip())
            risc = r[" RISC processor type"].strip()
            ev[(core, risc)].append(
                (int(r[" time[cycles since reset]"].strip()), r[" type"].strip())
            )
    per_core = defaultdict(int)
    for (core, _risc), lst in ev.items():
        lst.sort()
        st = None
        for t, typ in lst:
            if typ == "ZONE_START":
                st = t
            elif typ == "ZONE_END" and st is not None:
                per_core[core] = max(per_core[core], t - st)
                st = None
    if not per_core:
        return None, 0
    vals = sorted(per_core.values())
    return vals[-1], len(per_core)  # max per-core ns, core count


def run_config(n, nc, g, timeout=200):
    env = dict(os.environ)
    env["TT_METAL_DEVICE_PROFILER"] = "1"
    if os.path.exists(LOG):
        os.remove(LOG)
    p = subprocess.run(
        [
            "timeout",
            "-s",
            "KILL",
            str(timeout),
            "python3",
            os.path.join(HERE, "probe.py"),
            "softmax",
            str(n),
            str(nc),
            str(g),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    pcc = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            pcc = line.strip()
    ns, cores = per_core_kernel_ns()
    return pcc, ns, cores


def main():
    # (label, N, NC, G, tiles_per_core)
    configs = [
        ("1 tile/core, 1x1 (1 core)", 1, 1, 1, 1),
        ("1 tile/core, 1x64 (64 cores)", 64, 1, 64, 1),
        ("4 tiles/core, 64 cores", 128, 2, 64, 4),
        ("9 tiles/core, 64 cores", 192, 3, 64, 9),
    ]
    results = {}
    for label, n, nc, g, tpc in configs:
        pcc, ns, cores = run_config(n, nc, g)
        results[label] = (ns, cores, tpc)
        print(f"{label:32} cores={cores:>3} per-core softmax = {ns} ns   ({pcc})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: 1 tile/core, 1 core vs 64 cores
    a = ax[0]
    labs = ["1x1\n(1 core)", "1x64\n(64 cores)"]
    keys = ["1 tile/core, 1x1 (1 core)", "1 tile/core, 1x64 (64 cores)"]
    vals = [results[k][0] or 0 for k in keys]
    bars = a.bar(labs, vals, color=["#888", "C0"])
    for bar, v, k in zip(bars, vals, keys):
        a.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v}\n({results[k][1]} cores)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    a.set_ylabel("per-core softmax compute (ns)")
    a.set_title(
        "A. 1 tile/core: 1 core vs 64 cores\n(flat = parallelism, no per-core penalty)"
    )

    # Panel B: tiles/core scaling on 64 cores
    b = ax[1]
    sweep = [
        ("4 tiles/core, 64 cores", 4),
        ("9 tiles/core, 64 cores", 9),
        ("1 tile/core, 1x64 (64 cores)", 1),
    ]
    pts = sorted([(tpc, results[k][0]) for k, tpc in sweep if results[k][0]])
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        b.plot(xs, ys, "o-", color="C3")
        for x, y in zip(xs, ys):
            b.text(x, y, f"  {y}ns", fontsize=9, va="bottom")
    b.set_xlabel("tiles per core")
    b.set_ylabel("per-core softmax compute (ns)")
    b.set_title("B. d2m softmax per-core compute vs tiles/core (64-core virtual grid)")
    b.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "vg_scaling.png")
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
