# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-thread, span-validated softmax compute-kernel breakdown: d2m vs ttnn.

Both kernels instrumented with DeviceZoneScopedN and analyzed PER-THREAD (zones are
sequential within a thread = real wall-time, not a cross-thread sum). Each thread's
zone times sum to its TRISC-KERNEL span (validated, ~100% coverage).

d2m   : auto-pass (insert-device-zone-scopes, op-level zones), single-core [3,3].
ttnn  : hand-added phase zones in the (bundled) attention compute kernel.
BOTH matched: 6144x96 tensor, 64 cores, 3 tile-rows x 3 wide = 9 tiles/core.
Math-thread spans: d2m 59602 cyc, ttnn 18539 cyc (1GHz) -> d2m ~3.2x per-core here;
uninstrumented op-level (ttrt perf) was d2m 50895 / ttnn 28737 ns = 1.77x.

Geometry IS matched (same 6144x96 tensor, 64 cores, 9 tiles/core both sides; d2m via
the 64x1->8x8 virtual grid, ttnn via its native row-split). Remaining caveat: zone
granularity differs (d2m per-OP via the auto-pass; ttnn per-PHASE hand-added, so a
ttnn phase zone includes its own inits -> ttnn "productive" is if anything over-
counted, making the overhead gap conservative). Comparable metrics: per-core span
and the overhead fraction (inits + CB-sync/loop/barriers vs productive tile work).

Numbers below are the measured per-thread exclusive-time breakdowns (cycles, 1GHz).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# --- d2m single-core [3,3], per-thread (% of that thread's TRISC-KERNEL span) ---
# measured: TRISC_0/1/2 spans 59813/59667/59525; coverage ~100%
D2M = {
    "TRISC_1 (math)": {
        "reduce": 1,
        "exp": 34,
        "sub/div/bcast": 3,
        "fill(scaler)": 9,
        "init/reconfig": 10,
        "wrapper(sync/loop)": 42,
    },
    "TRISC_0 (unpack)": {
        "init/reconfig": 40,
        "wrapper(sync/loop)": 56,
        "exp": 0,
        "reduce": 1,
        "sub/div/bcast": 1,
    },
    "TRISC_2 (pack)": {"init/reconfig": 19, "wrapper(sync/loop)": 78, "reduce": 2},
}
# --- ttnn 3x3 attention kernel, core(1,1), per-thread (% of span) ---
# measured spans TRISC_0/1/2 = 6201/6322/6413
TTNN = (
    {  # matched: 6144x96 (192 tile-rows x 3), 64 cores, 3 rows x 3 wide = 9 tiles/core
        "TRISC_1 (math)": {"reduce": 68, "subexp": 18, "mul": 6, "wrapper/init": 8},
        "TRISC_0 (unpack)": {"reduce": 27, "subexp": 3, "mul": 3, "wrapper/init": 67},
        "TRISC_2 (pack)": {"reduce": 9, "subexp": 40, "mul": 8, "wrapper/init": 43},
    }
)


def overhead_frac(d):
    return sum(v for k, v in d.items() if "init" in k or "wrapper" in k)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: overhead fraction per thread (the comparable metric)
    a = ax[0]
    threads = ["TRISC_1 (math)", "TRISC_0 (unpack)", "TRISC_2 (pack)"]
    d2m_oh = [overhead_frac(D2M[t]) for t in threads]
    ttnn_oh = [overhead_frac(TTNN[t]) for t in threads]
    x = np.arange(len(threads))
    w = 0.38
    a.bar(x - w / 2, d2m_oh, w, label="d2m", color="C3")
    a.bar(x + w / 2, ttnn_oh, w, label="ttnn", color="C0")
    for i in range(len(threads)):
        a.text(
            x[i] - w / 2,
            d2m_oh[i],
            f"{d2m_oh[i]}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        a.text(
            x[i] + w / 2,
            ttnn_oh[i],
            f"{ttnn_oh[i]}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    a.set_xticks(x)
    a.set_xticklabels([t.split()[1] for t in threads])
    a.set_ylabel("overhead = init/reconfig + CB-sync/loop/barriers (% of thread span)")
    a.set_title(
        "Compute-thread OVERHEAD fraction (the comparable metric)\n"
        "d2m's per-op-pass structure vs ttnn's fused phases"
    )
    a.set_ylim(0, 100)
    a.legend()

    # Panel B: math-thread (TRISC_1) stacked breakdown
    b = ax[1]
    for i, (name, d) in enumerate(
        [("d2m", D2M["TRISC_1 (math)"]), ("ttnn", TTNN["TRISC_1 (math)"])]
    ):
        bottom = 0
        order = sorted(
            d.items(), key=lambda kv: ("wrapper" in kv[0] or "init" in kv[0], -kv[1])
        )
        for k, v in order:
            isoh = "wrapper" in k or "init" in k
            b.bar(
                i,
                v,
                bottom=bottom,
                color=("0.6" if isoh else "C2"),
                edgecolor="k",
                linewidth=0.4,
            )
            if v >= 4:
                b.text(
                    i, bottom + v / 2, f"{k} {v}%", ha="center", va="center", fontsize=7
                )
            bottom += v
    b.set_xticks([0, 1])
    b.set_xticklabels(["d2m", "ttnn"])
    b.set_ylabel("% of math-thread (TRISC_1) span")
    b.set_title(
        "Math-thread breakdown\nd2m ~47% productive / ~52% overhead;  ttnn ~92% / ~8%"
    )
    b.set_ylim(0, 105)

    fig.tight_layout()
    out = os.path.join(HERE, "kernel_breakdown.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    print(
        "\ncompute-thread (math) overhead: d2m =",
        overhead_frac(D2M["TRISC_1 (math)"]),
        "%   ttnn =",
        overhead_frac(TTNN["TRISC_1 (math)"]),
        "%",
    )


if __name__ == "__main__":
    main()
