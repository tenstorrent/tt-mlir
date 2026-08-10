# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Autotune the ``mesh_sigmoid`` KernelBench from ``test_mesh.py``.

The mesh bench's materializer declares a (1, 2) device mesh, shards the full
input across both devices, and gathers the result back, so the autotuner
sweeps per-shard grid/block/mem knobs while every config runs on the full
mesh.  Two caveats are specific to mesh benches (see the bench's comments):

* Auto-generated grid/block candidates are derived from the bench's FULL
  tensor shapes, but the kernel executes on per-device shards — pass explicit
  ``AutotuneKnobs.grid_shapes`` valid for the shard, as done here.
* ``kernel_ns`` from a multi-chip profiler trace is unreliable (perf-analyzer
  does not separate per-device cycle clocks), so this test asserts
  correctness (PCC) and completion, never timing.
"""

import pytest

# `import autotuner` inserts test/d2m-jit into sys.path (for runner.py), which
# also makes `test_mesh` importable here — keep this import first.
import autotuner as A
import test_mesh
from test_mesh import KERNEL_BENCHES, requires_mesh

pytestmark = pytest.mark.device_only(reason="autotuner is a silicon-only feature")

# Per-device shard is (128, 128) = 4x4 tiles with block [1, 1]; both grids
# divide the shard's block counts, so every config is feasible.
_GRIDS = [(1, 1), (2, 2)]


def test_generate_configs_mesh_bench_explicit_grids():
    """Pure config-space check: explicit per-shard grids pass through
    unchanged and combine with the bench's default block/mem (focused mode).
    No silicon needed."""
    bench = KERNEL_BENCHES["mesh_sigmoid"]
    knobs = A.AutotuneKnobs(grid_shapes=_GRIDS)
    cfgs = A.Autotuner(knobs=knobs, verbose=False).generate_configs(bench)
    assert [c.id for c in cfgs] == ["g1x1_b1x1_mL1", "g2x2_b1x1_mL1"]


@requires_mesh
def test_autotune_mesh_sigmoid_on_device(tmp_path):
    """End-to-end autotune of the mesh bench on a real 1x2 mesh.

    Asserts the device contract per config: the swept knob reached a
    constructed Layout (``_verify_config_applied``) and the gathered full
    tensor passes PCC against the golden — ``error is None`` means both held.
    Timing is deliberately not asserted (multi-chip ``kernel_ns`` caveat
    above, plus run-to-run variance).
    """
    bench = KERNEL_BENCHES["mesh_sigmoid"]
    tuner = A.Autotuner(
        knobs=A.AutotuneKnobs(grid_shapes=_GRIDS),
        # output_dir=str(tmp_path),
        check_pcc=True,
        n_warmup=0,
        verbose=False,
    )
    results = tuner.run_bench(bench, bench_name="mesh_sigmoid")
    tuner.save_results("mesh_sigmoid", results)
    tuner.save_summary({"mesh_sigmoid": results})

    assert len(results) == len(_GRIDS)
    for result in results:
        assert result.error is None, f"config {result.config_id} failed: {result.error}"
        assert result.pcc is not None
        assert result.pcc >= bench.pcc, f"PCC {result.pcc} < {bench.pcc}"


@requires_mesh
def test_autotune_kernel_mesh_sigmoid_on_device(tmp_path):
    """Same end-to-end mesh autotune as above, but through the high-level
    ``autotune_kernel`` entry point: load ``test_mesh.py`` as a kernel file,
    sweep the explicit per-shard grids, and save artifacts — all in one call.
    Assertions match ``test_autotune_mesh_sigmoid_on_device`` (correctness
    only, no timing)."""
    all_results = A.autotune_kernel(
        test_mesh.__file__,
        bench_names=["mesh_sigmoid"],
        # knobs=A.AutotuneKnobs(grid_shapes=[(1, 1), (2, 2), (1, 8)], mem_spaces="all"),#_GRIDS),
        # knobs=A.AutotuneKnobs(max_cores=4, max_block_tiles=4),
        # output_dir=str(tmp_path),
        check_pcc=True,
        n_warmup=0,
        # verbose=False,
    )

    assert list(all_results) == ["mesh_sigmoid"]
    results = all_results["mesh_sigmoid"]
    # assert len(results) == len(_GRIDS)
    bench = KERNEL_BENCHES["mesh_sigmoid"]
    for result in results:
        assert result.error is None, f"config {result.config_id} failed: {result.error}"
        assert result.pcc is not None
        assert result.pcc >= bench.pcc, f"PCC {result.pcc} < {bench.pcc}"
