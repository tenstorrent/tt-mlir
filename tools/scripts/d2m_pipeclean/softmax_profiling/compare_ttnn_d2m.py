# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn vs d2m softmax, true 1:1 (matched tensor, matched 64 physical cores).

Per-core softmax compute time (ns), all working configs. Both stacks run the same
tensor on 64 cores: ttnn splits tile-rows across the grid; d2m uses the 64x1->8x8
virtual grid. d2m can't yet do non-square rows/core (the [4,2]/[2,4] kernel-compile
hang), so we sweep square per-core blocks: 1, 4, 9 tiles/core.

ttnn numbers: ttrt perf DEVICE KERNEL DURATION (SoftmaxDeviceOperation).
d2m numbers : per-core longest TRISC-KERNEL interval from profile_log_device.csv
              (ttrt can't load the virtual-grid .ttm; run via d2m_jit + profiler).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# tiles/core -> (tensor tile shape RxW, ttnn ns, d2m ns), all on 64 cores
DATA = {
    1: ("64x1", 7452, 9858),
    4: ("128x2", 18717, 26009),
    9: ("192x3", 28737, 50895),
}


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tpc = sorted(DATA)
    ttnn = [DATA[t][1] for t in tpc]
    d2m = [DATA[t][2] for t in tpc]
    ratio = [d / n for d, n in zip(d2m, ttnn)]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: grouped bars, per-core compute
    a = ax[0]
    x = np.arange(len(tpc))
    w = 0.38
    a.bar(x - w / 2, ttnn, w, label="ttnn", color="C0")
    a.bar(x + w / 2, d2m, w, label="d2m (virtual 64-grid)", color="C3")
    for i, t in enumerate(tpc):
        a.text(
            x[i] - w / 2, ttnn[i], f"{ttnn[i]}", ha="center", va="bottom", fontsize=8
        )
        a.text(x[i] + w / 2, d2m[i], f"{d2m[i]}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x)
    a.set_xticklabels([f"{t}\n({DATA[t][0]})" for t in tpc])
    a.set_xlabel("tiles per core  (tensor tile shape)")
    a.set_ylabel("per-core softmax compute (ns)")
    a.set_title(
        "ttnn vs d2m softmax — matched tensor, 64 cores\n(true 1:1, per-core compute)"
    )
    a.legend()

    # Panel B: the ratio (the per-core kernel-efficiency gap)
    b = ax[1]
    b.plot(tpc, ratio, "o-", color="C2", linewidth=2)
    for t, r in zip(tpc, ratio):
        b.text(t, r, f"  {r:.2f}x", fontsize=10, va="bottom")
    b.axhline(1.0, color="gray", ls="--", lw=1, label="parity")
    b.set_xlabel("tiles per core")
    b.set_ylabel("d2m / ttnn  (per-core compute)")
    b.set_title(
        "Per-core kernel-efficiency gap grows with tiles/core\n"
        "(2D-loop materialization vs dst-register fusion)"
    )
    b.set_ylim(0, 2.0)
    b.set_xticks(tpc)
    b.grid(alpha=0.3)
    b.legend()

    fig.tight_layout()
    out = os.path.join(HERE, "compare_ttnn_d2m.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    print("\ntiles/core | ttnn ns | d2m ns | d2m/ttnn")
    for t in tpc:
        print(
            f"    {t:>2}     | {DATA[t][1]:>7} | {DATA[t][2]:>6} | {ratio[tpc.index(t)]:.2f}x"
        )


if __name__ == "__main__":
    main()
