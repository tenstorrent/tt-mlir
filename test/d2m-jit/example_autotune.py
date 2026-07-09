# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example: call the autotuner programmatically.

Equivalent CLI commands are shown in the comments below.

Run this script from the repo root after activating the venv:
    source env/activate
    python3 test/d2m-jit/example_autotune.py
"""

import sys

sys.path.insert(0, "test/d2m-jit")

from autotuner import Autotuner, AutotuneKnobs, autotune_kernel, load_kernel_module

# ---------------------------------------------------------------------------
# Example 1: full auto-sweep with default heuristics
#
# Equivalent CLI:
#   python3 test/d2m-jit/autotuner.py \
#       --kernel test/d2m-jit/kernels/prefill/rope.py \
#       --bench rope \
#       --mem-spaces L1,DRAM \
#       --check-pcc \
#       --output-dir autotune-artifacts
# ---------------------------------------------------------------------------


def a():
    results = autotune_kernel(
        "test/d2m-jit/kernels/prefill/rope.py",
        bench_names=["rope"],
        # knobs=AutotuneKnobs(mem_spaces=["L1", "DRAM"]),
        check_pcc=True,
        output_dir="autotune-artifacts",
    )

    best = min(
        (r for r in results["rope"] if r.kernel_ns is not None),
        key=lambda r: r.kernel_ns,
    )
    print(f"Best config: {best.config_id}  →  {best.kernel_ns:.1f} ns")


# ---------------------------------------------------------------------------
# Example 2: constrained sweep — specific grids, blocks, and mem spaces
#
# Equivalent CLI:
#   python3 test/d2m-jit/autotuner.py \
#       --kernel test/d2m-jit/kernels/prefill/rope.py \
#       --bench rope \
#       --grid-shapes 1x1,2x2,4x4 \
#       --block-shapes 1x1,2x2 \
#       --mem-spaces L1,DRAM \
#       --n-warmup 2 \
#       --save-profiler-logs \
#       --output-dir autotune-artifacts/constrained
# ---------------------------------------------------------------------------


def b():
    knobs = AutotuneKnobs(
        grid_shapes=[(1, 1), (2, 2), (4, 4)],
        block_shapes=[[1, 1], [2, 2]],
        mem_spaces=["L1", "DRAM"],
    )

    tuner = Autotuner(
        knobs=knobs,
        output_dir="autotune-artifacts/constrained",
        save_profiler_logs=True,
        check_pcc=True,
        n_warmup=2,
    )

    mod = load_kernel_module("test/d2m-jit/kernels/prefill/rope.py")
    bench = mod.KERNEL_BENCHES["rope"]

    results2 = tuner.run_bench(bench, bench_name="rope")
    tuner.save_results("rope", results2)
    tuner.save_summary({"rope": results2})


# ---------------------------------------------------------------------------
# Example 3: single default-config run (no sweep) — quick correctness check
#
# Equivalent CLI:
#   python3 test/d2m-jit/autotuner.py \
#       --kernel test/d2m-jit/kernels/prefill/rope.py \
#       --no-sweep \
#       --check-pcc \
#       --n-warmup 0 \
#       --output-dir autotune-artifacts/default
# ---------------------------------------------------------------------------


def c():
    from autotuner import AutotuneConfig

    default_ts = bench.tensors[0]
    default_cfg = AutotuneConfig(
        grid_shape=tuple(bench.grid_shape),
        block_shape=list(default_ts.block_shape),
        mem_space="L1",
    )

    single_tuner = Autotuner(
        output_dir="autotune-artifacts/default",
        check_pcc=True,
        n_warmup=0,
    )
    result = single_tuner.run_config(bench, default_cfg, bench_name="rope")
    single_tuner.save_results("rope", [result])
    single_tuner.save_summary({"rope": [result]})
    print(
        f"Default config: {result.config_id}  →  {result.kernel_ns} ns  pcc={result.pcc}"
    )


a()
