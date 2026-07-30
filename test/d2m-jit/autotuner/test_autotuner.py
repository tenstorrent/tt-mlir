# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the d2m-jit autotuner.

Most of these cover the autotuner's pure decision logic — config-space
enumeration, the grid/block divisor heuristics, config identity, CLI argument
parsing, result ranking, and tensor overrides — none of which touch silicon,
so they run fast and deterministically.

``test_autotune_exp_on_device`` is the one on-silicon smoke test: it runs the
autotuner end-to-end against a real KernelBench and checks the device contract
the pure tests can't reach — that a swept knob actually reached a constructed
Layout, and that the resulting numerics pass PCC.
"""

import pathlib

import pytest
import torch

import autotuner as A
from runner import KernelBench, TensorSpec

_KERNELS_DIR = pathlib.Path(__file__).parent.parent / "kernels"


def _bench(shapes, block_shape=None, dtype=torch.float32, grid_shape=(1, 1)):
    """Build a KernelBench with dummy callables.

    generate_configs / valid_* never invoke the kernel/golden/run callables,
    so dummies are enough to exercise the pure config-generation paths.
    """
    tensors = [
        TensorSpec(shape=s, block_shape=list(block_shape or [1, 1]), dtype=dtype)
        for s in shapes
    ]
    return KernelBench(
        kernel=lambda *a, **k: None,
        golden=lambda *a, **k: None,
        run=lambda *a, **k: None,
        tensors=tensors,
        grid_shape=grid_shape,
        name="fake",
    )


def _tuner(knobs):
    return A.Autotuner(knobs=knobs, verbose=False)


# ---------------------------------------------------------------------------
# Divisor / grid / block heuristics
# ---------------------------------------------------------------------------


def test_divisors():
    assert A._divisors(12) == [1, 2, 3, 4, 6, 12]
    assert A._divisors(1) == [1]
    assert A._divisors(0) == []


def test_valid_grid_shapes_divide_tile_dims():
    # 128x128 -> 4x4 tiles; every returned grid must divide (4, 4).
    bench = _bench([(128, 128)])
    grids = A.valid_grid_shapes(bench, A.AutotuneKnobs())
    assert (1, 1) in grids
    assert (2, 2) in grids
    for gy, gx in grids:
        assert 4 % gy == 0 and 4 % gx == 0
        assert gy * gx <= 8  # default max_cores


def test_valid_grid_shapes_respects_max_cores():
    bench = _bench([(256, 256)])  # 8x8 tiles
    grids = A.valid_grid_shapes(bench, A.AutotuneKnobs(max_cores=4))
    assert grids  # non-empty
    assert all(gy * gx <= 4 for gy, gx in grids)


def test_valid_grid_shapes_explicit_passthrough():
    bench = _bench([(128, 128)])
    explicit = [(1, 1), (2, 2)]
    assert A.valid_grid_shapes(bench, A.AutotuneKnobs(grid_shapes=explicit)) == explicit


def test_valid_block_shapes_divide_per_core_tiles():
    # 128x128 -> 4x4 tiles; on a 2x2 grid each core owns 2x2 tiles.
    bench = _bench([(128, 128)])
    blocks = A.valid_block_shapes(bench, (2, 2), A.AutotuneKnobs())
    assert [1, 1] in blocks
    assert [2, 2] in blocks
    for by, bx in blocks:
        assert 2 % by == 0 and 2 % bx == 0


def test_valid_block_shapes_gcd_across_tensors():
    # Two tensors with different x tile counts (4 and 2) -> gcd 2, so bx <= 2.
    bench = _bench([(32, 128), (32, 64)])
    blocks = A.valid_block_shapes(bench, (1, 1), A.AutotuneKnobs())
    assert all(bx in (1, 2) for _, bx in blocks)


# ---------------------------------------------------------------------------
# AutotuneConfig identity / uniform / serialization
# ---------------------------------------------------------------------------


def test_config_uniform_broadcasts():
    cfg = A.AutotuneConfig.uniform((2, 2), [1, 1], "L1", n_tensors=3)
    assert len(cfg.blocks) == 3
    assert len(cfg.mems) == 3
    assert cfg.is_uniform
    assert cfg.id == "g2x2_b1x1_mL1"


def test_config_id_collapses_uniform_axes():
    uniform = A.AutotuneConfig((1, 1), [[1, 1], [1, 1]], ["L1", "L1"])
    mixed = A.AutotuneConfig((1, 1), [[1, 1], [2, 2]], ["L1", "DRAM"])
    assert uniform.id == "g1x1_b1x1_mL1"
    assert mixed.id == "g1x1_b1x1-2x2_mL1-DRAM"
    assert not mixed.is_uniform


def test_config_as_dict_roundtrips_fields():
    cfg = A.AutotuneConfig((2, 1), [[1, 2]], ["DRAM"])
    d = cfg.as_dict()
    assert d == {"grid_shape": [2, 1], "blocks": [[1, 2]], "mems": ["DRAM"]}


# ---------------------------------------------------------------------------
# Config-space generation
# ---------------------------------------------------------------------------


def test_generate_configs_full_sweep_nonempty_and_unique():
    bench = _bench([(128, 128)])
    cfgs = _tuner(A.AutotuneKnobs()).generate_configs(bench)
    assert cfgs
    ids = [c.id for c in cfgs]
    assert len(ids) == len(set(ids)), "full-sweep produced duplicate config ids"
    # Full sweep tries both mem spaces.
    assert any(c.mems == ["L1"] for c in cfgs)
    assert any(c.mems == ["DRAM"] for c in cfgs)


def test_generate_configs_focused_grid_only():
    # A single explicit knob -> focused mode: other axes stay at bench defaults.
    bench = _bench([(128, 128)])
    knobs = A.AutotuneKnobs(grid_shapes=[(1, 1), (2, 2)])
    cfgs = _tuner(knobs).generate_configs(bench)
    assert [c.id for c in cfgs] == ["g1x1_b1x1_mL1", "g2x2_b1x1_mL1"]


def test_generate_configs_joint_mem_spaces_all():
    bench = _bench([(64, 64), (64, 64)])
    knobs = A.AutotuneKnobs(joint_mem_spaces="all", grid_shapes=[(1, 1)])
    cfgs = _tuner(knobs).generate_configs(bench)
    mems = {tuple(c.mems) for c in cfgs}
    assert mems == {
        ("L1", "L1"),
        ("L1", "DRAM"),
        ("DRAM", "L1"),
        ("DRAM", "DRAM"),
    }


def test_generate_configs_per_tensor_count_matches_bench():
    bench = _bench([(64, 64), (64, 64), (64, 64)])
    cfgs = _tuner(A.AutotuneKnobs()).generate_configs(bench)
    assert cfgs
    for c in cfgs:
        assert len(c.blocks) == 3
        assert len(c.mems) == 3


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_parse_shape_and_grid():
    assert A._parse_shape("2x4") == [2, 4]
    assert A._parse_grids("1x1,2x2") == [(1, 1), (2, 2)]
    assert A._parse_shapes("1x1,2x2") == [[1, 1], [2, 2]]


def test_parse_shape_rejects_malformed():
    with pytest.raises(Exception):
        A._parse_shape("123")


def test_parse_tensor_shapes():
    assert A._parse_tensor_shapes("64x96,96x64") == [(64, 96), (96, 64)]


def test_parse_joint_mems():
    assert A._parse_joint_mems("all") == "all"
    assert A._parse_joint_mems("L1-L1,L1-DRAM") == [["L1", "L1"], ["L1", "DRAM"]]


def test_parse_joint_blocks():
    assert A._parse_joint_blocks("1x1_1x1,1x3_3x1") == [
        [[1, 1], [1, 1]],
        [[1, 3], [3, 1]],
    ]


def test_parse_dtypes_aliases():
    assert A._parse_dtypes("bf16,float32") == [torch.bfloat16, torch.float32]
    with pytest.raises(Exception):
        A._parse_dtypes("float8")


# ---------------------------------------------------------------------------
# Result ranking
# ---------------------------------------------------------------------------


def _result(cfg_id_grid, kernel_ns, error=None, pcc=None):
    cfg = A.AutotuneConfig(cfg_id_grid, [[1, 1]], ["L1"])
    return A.AutotuneResult(
        bench_name="fake",
        config=cfg,
        kernel_ns=kernel_ns,
        pcc=pcc,
        error=error,
    )


def test_rank_sorts_valid_and_excludes_failures():
    slow = _result((1, 1), 300.0)
    fast = _result((2, 1), 100.0)
    mid = _result((4, 1), 200.0)
    errored = _result((1, 2), 50.0, error="boom")  # fast but errored -> excluded
    no_data = _result((2, 2), None)  # no profiler data -> excluded

    valid, failed = A._rank([slow, fast, mid, errored, no_data])

    assert [r.kernel_ns for r in valid] == [100.0, 200.0, 300.0]
    assert errored in failed
    assert no_data in failed
    assert A._best_result([slow, fast, mid, errored, no_data]) is fast


# ---------------------------------------------------------------------------
# Bench tensor overrides
# ---------------------------------------------------------------------------


def test_override_bench_tensors_shapes_and_dtypes():
    bench = _bench([(64, 64), (64, 64)])
    out = A._override_bench_tensors(
        bench,
        tensor_shapes=[(128, 128), (256, 256)],
        tensor_dtypes=[torch.bfloat16, torch.float32],
    )
    assert [t.shape for t in out.tensors] == [(128, 128), (256, 256)]
    assert [t.dtype for t in out.tensors] == [torch.bfloat16, torch.float32]
    # Original bench is untouched (dataclasses.replace returns a copy).
    assert bench.tensors[0].shape == (64, 64)


def test_override_bench_tensors_length_mismatch_raises():
    bench = _bench([(64, 64), (64, 64)])
    with pytest.raises(ValueError):
        A._override_bench_tensors(bench, tensor_shapes=[(128, 128)])


def test_override_bench_tensors_noop_returns_same():
    bench = _bench([(64, 64)])
    assert A._override_bench_tensors(bench) is bench


# ---------------------------------------------------------------------------
# On-silicon smoke test
# ---------------------------------------------------------------------------


def test_autotune_exp_on_device(tmp_path):
    """End-to-end autotune of the `exp` KernelBench on device.

    This is the one test here that needs silicon.  It runs a real bench
    through the autotuner and asserts the device contract the pure tests
    cannot: with ``check_pcc``, ``run_config`` sets ``error`` if the swept
    knob was never applied (``_verify_config_applied``) or if the output
    fails PCC, so ``error is None`` means both held.  Timing is deliberately
    not asserted — ``kernel_ns`` depends on the profiler build and varies
    run to run, which would make the test flaky.
    """
    kernel_file = _KERNELS_DIR / "patterns" / "eltwise_exp_to_kernel.py"
    all_results = A.autotune_kernel(
        str(kernel_file),
        bench_names=["exp"],
        knobs=A.AutotuneKnobs(grid_shapes=[(1, 1)]),  # single config: keep it quick
        output_dir=str(tmp_path),
        check_pcc=True,
        n_warmup=0,
        verbose=False,
        strategy="default",
    )

    assert "exp" in all_results
    results = all_results["exp"]
    assert len(results) == 1
    result = results[0]
    assert result.error is None, f"autotune run failed: {result.error}"
    assert result.pcc is not None
    mod = A.load_kernel_module(str(kernel_file))
    bench = mod.KERNEL_BENCHES["exp"]
    assert result.pcc >= bench.pcc, f"PCC {result.pcc} < {bench.pcc}"
