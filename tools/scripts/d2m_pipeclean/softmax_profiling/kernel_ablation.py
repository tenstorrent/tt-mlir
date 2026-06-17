# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hand-edit ablation of the d2m softmax MATH kernel (single-core [3,3], fp32).

Dumped the generated TRISC math kernel (kernel5) via `ttrt perf --dump-kernels`,
hand-edited it, and re-timed each change with `ttrt perf --load-kernels`
(DEVICE KERNEL DURATION). Each step is cumulative on the prior one.

Correctness: the final (A3) kernel was run with `ttrt run --load-kernels --seed 0
--save-artifacts`; its device output PCC=1.0 vs torch.softmax(dim=1), row-sums in
[0.9984, 1.0012]. A1/A2 are semantics-preserving by construction; A3 reproduces
ttnn's own recip-in-reduce + bcast-mul numerics (moreh_softmax_w.cpp:168-206).

What each step does:
  A1  dedupe 10 exact-duplicate top inits (compute_kernel_hw_startup / init_sfpu)
      + hoist reduce_init/reduce_uninit out of the per-tile reduce body
      (init-once / reduce-many / uninit-once, the standard reduce idiom).
  A2  fuse sub + exp in DST: exp the (x-max) result in-register and pack straight
      to the exps CB -> kills the separate exp pass, its L1 round-trip, and a
      barrier. Small here: on one uncontended core the L1 hop is cheap.
  A3  fold recip into the sum-reduce (1/sum computed in-DST before pack) and swap
      the per-tile bcast `div` for a bcast `mul`. Biggest single win: SFPU
      division is iterative/expensive vs reciprocal-once-per-row + multiply.

ttnn per-core reference for the same 9 tiles ~= 28737 ns (matched 64-core op-level
DEVICE KERNEL DURATION). The edits close the gap from 1.77x -> 1.41x. The residual
is structural: 5 remaining L1-materializing passes (fill-scalers, reduce_max,
exps, reduce_sum) each gated by a load-bearing unpack_stall_on_pack barrier, plus
per-tile op-type reconfig. Removing those needs true cross-op loop fusion (keep
intermediates in DST across reduce->sub->exp->reduce->mul), i.e. a compiler pass,
not a line edit -- the barriers cannot simply be deleted (the loops write via
pack_tile<true> direct-to-L1, so ordering depends on them; deleting corrupts).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# (label, cumulative ns, note)
STEPS = [
    ("baseline", 51024, "generated d2m [3,3] math kernel"),
    ("A1 fold inits", 47619, "dedupe top inits + hoist reduce init/uninit"),
    ("A2 fuse sub+exp", 46972, "exp in-DST, kill exp pass + L1 hop + barrier"),
    ("A3 recip+mul", 40399, "fold recip into reduce, div -> mul  (PCC=1.0)"),
]
TTNN_REF = 28737  # per-core, 9 tiles/core, matched 64-core op-level


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [s[0] for s in STEPS]
    ns = [s[1] for s in STEPS]
    x = np.arange(len(STEPS))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, ns, color=["0.5", "C0", "C0", "C2"], edgecolor="k", linewidth=0.5)
    ax.axhline(
        TTNN_REF,
        color="C3",
        ls="--",
        lw=1.5,
        label=f"ttnn per-core ref ({TTNN_REF} ns)",
    )
    base = ns[0]
    for i, v in enumerate(ns):
        d = 100.0 * (v - base) / base
        ax.text(i, v + 400, f"{v}\n{d:+.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(
            i,
            v / 2,
            f"{v / TTNN_REF:.2f}x",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("DEVICE KERNEL DURATION (ns), single-core [3,3]")
    ax.set_title(
        "Hand-edit ablation of d2m softmax math kernel\n"
        f"51024 -> 40399 ns (-20.8%); gap to ttnn 1.77x -> 1.41x"
    )
    ax.set_ylim(0, max(ns) * 1.15)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "kernel_ablation.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    print("\nstep              | ns    | d cum  | xttnn | note")
    for lbl, v, note in STEPS:
        print(
            f"{lbl:17s} | {v:5d} | {100*(v-base)/base:+5.1f}% | {v/TTNN_REF:.2f}x | {note}"
        )


if __name__ == "__main__":
    main()
