# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import OrderedDict
from typing import List, Optional, Sequence

import torch

import chisel
from ttmlir.dialects import ttnn
from builder.base.builder_apis import compile_and_execute_ttnn, compile_and_execute_ttir
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.ttir.ttir_builder import TTIRBuilder


# n300 mesh layout used by the multichip tests; the `multichip_device` fixture
# opens a matching (1, 2) mesh device and skips when the host has fewer chips.
_MULTICHIP_MESH = OrderedDict([("x", 1), ("y", 2)])

# Compile pipeline for tests that emit explicit distribute_tensor /
# aggregate_tensor ops. The default TTNN backend pipeline inserts its own
# to_device / to_layout passes which conflict with hand-rolled host-side
# distribute ops, so we mirror the minimal pipeline used by
# `test/python/golden/test_ttnn_ops.py::test_all_gather`.
_STYLE_A_PIPELINE = (
    "ttcore-mark-functions-as-forward,"
    "ttcore-wrap-device-module,"
    "ttcore.device_module(builtin.module("
    "ttnn-configure-ccl-ops,ttnn-deallocate))"
)

# Host-side input encoding required by `ttnn.distribute_tensor`: the source
# tensor must live in system memory in row-major layout before it can be split
# across the mesh.
_SYSTEM_MEM_RM = {
    "layout": ttnn.Layout.RowMajor,
    "buffer_type": ttnn.BufferType.SystemMemory,
}


def test_chisel_records_one_layer_nn(request, device, tmp_path):
    # multiply(y, y) instead of an activation: the TTNN optimizer fuses
    # activations into ttnn.linear, collapsing two ops into one.
    x_shape = (64, 128)
    w_shape = (256, 128)  # transpose_b=True -> effective rhs is (128, 256)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    ops_seen = {r.op for r in records}
    expected = {"ttnn.linear", "ttnn.multiply"}
    missing = expected - ops_seen
    assert not missing, (
        f"chisel report missing op records for: {sorted(missing)}. "
        f"Saw: {sorted(ops_seen)}"
    )

    PCC_THRESHOLD = 0.99

    def assert_pcc(mode: chisel.NumericsMode) -> None:
        label = str(mode)
        pcc_records = [
            r for r in records if r.check == "numerics" and r.payload.mode == mode
        ]
        assert pcc_records, f"no {label} (PCC) records produced"
        for op in expected:
            op_pcc = [r for r in pcc_records if r.op == op]
            assert op_pcc, f"no {label} PCC record for {op}"
            for r in op_pcc:
                assert (
                    r.status == chisel.RecordStatus.OK
                ), f"{op}: {label} PCC status={r.status} payload={r.payload}"
                assert (
                    r.payload.pcc >= PCC_THRESHOLD
                ), f"{op}: {label} PCC {r.payload.pcc} below threshold {PCC_THRESHOLD}"

    assert_pcc(chisel.NumericsMode.ISOLATED)
    assert_pcc(chisel.NumericsMode.ACCUMULATED)

    # One promotion per function arg (x, w, b); extras mean a golden is missing.
    EXPECTED_PROMOTIONS = 3
    promotions = [r for r in records if r.check == "golden_promoted"]
    assert len(promotions) == EXPECTED_PROMOTIONS, (
        f"expected {EXPECTED_PROMOTIONS} golden_promoted records "
        f"(one per function arg), got {len(promotions)}: "
        f"{[(r.op, r.ssa) for r in promotions]}"
    )


def test_chisel_dumps_debug_artifacts_on_pcc_fail(request, device, tmp_path):
    # Set the PCC threshold above the [-1, 1] range so every numerics check
    # records NUMERICS_FAIL, and verify the recorder dumps the MLIR source
    # and flatbuffer for the offending binary to debug_chisel_dir.
    x_shape = (64, 128)
    w_shape = (256, 128)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    debug_dir = tmp_path / "chisel_debug"

    with chisel.session(
        debug_chisel_dir=str(debug_dir),
        checks_config=chisel.ChiselChecksConfig(pcc=chisel.PCCConfig(min_pcc=2.0)),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    pcc_records = [r for r in records if r.check == "numerics"]
    assert pcc_records, "no numerics (PCC) records produced"
    assert all(
        r.status == chisel.RecordStatus.NUMERICS_FAIL for r in pcc_records
    ), f"expected all PCC records to fail with threshold=2.0, got {[r.status for r in pcc_records]}"

    assert debug_dir.is_dir(), f"debug dir not created at {debug_dir}"
    mlir_files = sorted(debug_dir.glob("binary_*.mlir"))
    fb_files = sorted(debug_dir.glob("binary_*.ttnn"))
    assert mlir_files, f"no MLIR dumped under {debug_dir}: {list(debug_dir.iterdir())}"
    assert (
        fb_files
    ), f"no flatbuffer dumped under {debug_dir}: {list(debug_dir.iterdir())}"
    # One dump per binary - the policy is first-failure-wins.
    assert len(mlir_files) == 1
    assert len(fb_files) == 1
    for mlir_path, fb_path in zip(mlir_files, fb_files):
        assert mlir_path.stat().st_size > 0, f"empty MLIR dump at {mlir_path}"
        assert fb_path.stat().st_size > 0, f"empty flatbuffer dump at {fb_path}"


def test_chisel_records_chisel_bug_on_callback_raise(
    request, device, tmp_path, monkeypatch
):
    # Monkey-patch the numerics check that _default_post_op calls so it
    # raises. chisel_safe should swallow the exception, write a chisel_bug
    # record carrying the traceback, and the ttmlir runtime should finish
    # the program normally so the test exits the session cleanly.
    import chisel.callbacks as chisel_callbacks

    boom_message = "synthetic chisel bug for testing chisel_safe"

    def boom(*args, **kwargs):
        raise RuntimeError(boom_message)

    monkeypatch.setattr(chisel_callbacks, "check_numerics", boom)

    x_shape = (64, 128)
    w_shape = (256, 128)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    with chisel.session() as report:
        # Must not raise - chisel_safe is supposed to keep the runtime alive
        # even when a chisel callback explodes.
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty - runtime did not produce any records"

    bug_records = [r for r in records if r.status == chisel.RecordStatus.CHISEL_BUG]
    assert bug_records, (
        "expected at least one chisel_bug record, got statuses: "
        f"{[r.status for r in records]}"
    )

    for r in bug_records:
        tb = r.payload.traceback
        assert tb, f"chisel_bug record has empty traceback for op={r.op}"
        assert (
            boom_message in tb
        ), f"traceback for op={r.op} missing the raised message; got:\n{tb}"
        assert (
            "RuntimeError" in tb
        ), f"traceback for op={r.op} missing the exception type; got:\n{tb}"

    bug_ops = {r.op for r in bug_records}
    assert bug_ops == {
        "ttnn.linear",
        "ttnn.multiply",
    }, f"expected chisel_bug for both ops, got {bug_ops}"


# ---------------------------------------------------------------------------
# Multichip variants
#
# Both tests below run a single matmul on an n300-style (1, 2) mesh. They use
# the same body but flip the sharding plan to exercise the two canonical
# parallelisms:
#
# - data-parallel: input is split along the batch dim, weight is replicated,
#   each chip runs an independent matmul on its slice, and the per-chip
#   results are concatenated back along the batch dim.
# - tensor-parallel: input is replicated, weight is split along the output
#   dim, each chip produces a partial output, and the per-chip results are
#   concatenated along the output dim.
#
# The sharding is written into the IR explicitly via `ttnn.distribute_tensor`
# (on the host SystemMemory inputs) and `ttnn.aggregate_tensor` (on the host
# result). chisel still records the underlying `ttnn.matmul` op once per
# device shard so the per-chip PCC accounting is exercised end-to-end.
# ---------------------------------------------------------------------------


def _build_matmul_module(
    builder: TTNNBuilder,
    x_shape: Sequence[int],
    w_shape: Sequence[int],
    x_shard_dims: List[int],
    w_shard_dims: List[int],
    out_shard_dims: List[int],
):
    """Author a single-matmul TTNN function with explicit distribute/aggregate.

    Each input arrives at host (SystemMemory, RowMajor), is distributed
    according to its `shard_dims` (`-1` => replicate on that mesh axis,
    `>=0` => shard along that tensor dim), then moved on-device and tilized
    for the matmul. The result is untilized, brought back to host, and
    aggregated using `out_shard_dims`.
    """

    @builder.func(
        [x_shape, w_shape],
        [torch.float32, torch.float32],
        custom_inputs=[_SYSTEM_MEM_RM, _SYSTEM_MEM_RM],
    )
    def matmul_program(
        x: Operand,
        w: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        device = builder.get_device()

        x_dist = builder.distribute_tensor(x, device=device, shard_dims=x_shard_dims)
        w_dist = builder.distribute_tensor(w, device=device, shard_dims=w_shard_dims)

        x_dev = builder.to_layout(
            builder.to_device(x_dist, device=device), layout=ttnn.Layout.Tile
        )
        w_dev = builder.to_layout(
            builder.to_device(w_dist, device=device), layout=ttnn.Layout.Tile
        )

        y = builder.matmul(x_dev, w_dev)

        y_rm = builder.to_layout(y, layout=ttnn.Layout.RowMajor)
        y_host = builder.from_device(y_rm)
        return builder.aggregate_tensor(
            y_host, device=device, shard_dims=out_shard_dims
        )


def _mesh_size(mesh: OrderedDict) -> int:
    size = 1
    for axis in mesh.values():
        size *= axis
    return size


def _assert_matmul_pcc(
    records,
    mesh: OrderedDict,
    pcc_threshold: float = 0.99,
) -> None:
    """Common PCC checks for the multichip matmul tests.

    For each numerics mode chisel emits, verifies that the `ttnn.matmul` op
    produced exactly one PCC record per chip in `mesh` (i.e. covers every
    `device_id` exactly once) and that every record passes the threshold.
    """
    assert records, "chisel report is empty"

    ops_seen = {r.op for r in records}
    assert (
        "ttnn.matmul" in ops_seen
    ), f"expected ttnn.matmul in chisel records, got {sorted(ops_seen)}"

    matmul_pcc = [r for r in records if r.op == "ttnn.matmul" and r.check == "numerics"]
    assert matmul_pcc, "no PCC records produced for ttnn.matmul"

    expected_chips = _mesh_size(mesh)
    expected_device_ids = set(range(expected_chips))
    modes_seen = {r.payload.mode for r in matmul_pcc}
    for mode in modes_seen:
        mode_records = [r for r in matmul_pcc if r.payload.mode == mode]
        assert len(mode_records) == expected_chips, (
            f"expected {expected_chips} ttnn.matmul PCC records in {mode} mode "
            f"(one per chip in mesh {dict(mesh)}), got {len(mode_records)}"
        )
        device_ids = {r.payload.device_id for r in mode_records}
        assert device_ids == expected_device_ids, (
            f"expected one ttnn.matmul PCC record per device_id in "
            f"{expected_device_ids} for {mode} mode, got {device_ids}"
        )

    for r in matmul_pcc:
        assert (
            r.status == chisel.RecordStatus.OK
        ), f"ttnn.matmul PCC status={r.status} payload={r.payload}"
        assert (
            r.payload.pcc >= pcc_threshold
        ), f"ttnn.matmul PCC {r.payload.pcc} below threshold {pcc_threshold}"


def test_chisel_records_matmul_data_parallel_multichip(
    request, multichip_device, tmp_path
):
    # Data-parallel: x is split along its batch dim (dim 0) across mesh
    # axis 1, w is replicated, each chip runs an independent (32, 128) @
    # (128, 256) matmul, and the per-chip (32, 256) outputs are concatenated
    # back along the batch dim into a single (64, 256) host tensor.
    x_shape = (64, 128)
    w_shape = (128, 256)

    def module(builder: TTNNBuilder):
        _build_matmul_module(
            builder,
            x_shape=x_shape,
            w_shape=w_shape,
            x_shard_dims=[-1, 0],
            w_shard_dims=[-1, -1],
            out_shard_dims=[-1, 0],
        )

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=multichip_device,
            mesh_dict=_MULTICHIP_MESH,
            custom_pipeline=_STYLE_A_PIPELINE,
        )
        records = report.records

    _assert_matmul_pcc(records, mesh=_MULTICHIP_MESH)


def test_chisel_records_matmul_tensor_parallel_multichip(
    request, multichip_device, tmp_path
):
    # Tensor-parallel: x is replicated across the (1, 2) mesh, w is split
    # along its output dim (dim 1) across mesh axis 1, each chip computes a
    # partial (64, 128) output, and aggregate_tensor concatenates the two
    # chunks back along the output dim into a single (64, 256) host tensor.
    x_shape = (64, 128)
    w_shape = (128, 256)

    def module(builder: TTNNBuilder):
        _build_matmul_module(
            builder,
            x_shape=x_shape,
            w_shape=w_shape,
            x_shard_dims=[-1, -1],
            w_shard_dims=[-1, 1],
            out_shard_dims=[-1, 1],
        )

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=multichip_device,
            mesh_dict=_MULTICHIP_MESH,
            custom_pipeline=_STYLE_A_PIPELINE,
        )
        records = report.records

    _assert_matmul_pcc(records, mesh=_MULTICHIP_MESH)


def test_chisel_records_update_cache_inplace(request, device, tmp_path):
    # ttir.update_cache canonicalizes to ttir.paged_update_cache (see
    # UpdateCacheOp::getCanonicalizationPatterns in TTIROps.cpp), which lowers
    # to ttnn.paged_update_cache - an in-place op that mutates `cache` and has
    # no SSA result, so every numerics record on it describes the in-place
    # cache operand.
    cache_shape = (1, 32, 64, 512)
    update_shape = (1, 32, 1, 512)
    index_shape = (1,)

    def module(builder: TTIRBuilder):
        @builder.func(
            [cache_shape, update_shape, index_shape],
            [torch.float32, torch.float32, torch.int32],
        )
        def update_cache(
            in0: Operand,  # cache
            in1: Operand,  # update values
            in2: Operand,  # update index
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            update_index = torch.randint(
                0, cache_shape[2], index_shape, dtype=torch.int32
            )
            builder.set_goldens(inputs={in2: update_index})
            return builder.update_cache(in0, in1, in2)

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:
        compile_and_execute_ttir(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    cache_numerics = [
        r
        for r in records
        if r.op == "ttnn.paged_update_cache" and r.check == "numerics"
    ]
    assert cache_numerics, (
        "no numerics records produced for ttnn.paged_update_cache; "
        f"saw: {[(r.op, r.check) for r in records]}"
    )

    PCC_THRESHOLD = 0.99
    for mode in (chisel.NumericsMode.ISOLATED, chisel.NumericsMode.ACCUMULATED):
        for_mode = [r for r in cache_numerics if r.payload.mode == mode]
        assert for_mode, f"no cache numerics record for mode={mode}"
        for r in for_mode:
            assert (
                r.status == chisel.RecordStatus.OK
            ), f"cache {mode} fail: payload={r.payload}"
            assert (
                r.payload.pcc >= PCC_THRESHOLD
            ), f"cache {mode} PCC {r.payload.pcc} below {PCC_THRESHOLD}"

    # When a chisel golden is registered, the pool entry for an in-place
    # operand is refreshed, not evicted.
    evictions = [
        r
        for r in records
        if r.check == "golden_evicted" and r.op == "ttnn.paged_update_cache"
    ]
    assert not evictions, (
        f"unexpected golden_evicted records on ttnn.paged_update_cache: "
        f"{[r.ssa for r in evictions]}"
    )


def test_chisel_records_batch_norm_training_inplace(request, device, tmp_path):
    # ttnn.batch_norm_training has one SSA result (the normalized output) plus
    # two in-place mutated operands (`running_mean`, `running_var`), so each
    # mode produces numerics records for the SSA result and both in-place
    # operands.
    n, c, h, w = 1, 4, 8, 8
    input_shape = (n, c, h, w)
    param_shape = (c,)

    def module(builder: TTIRBuilder):
        @builder.func(
            [input_shape, param_shape, param_shape, param_shape, param_shape],
            [torch.float32] * 5,
        )
        def batch_norm_training(
            in0: Operand,
            scale: Operand,
            offset: Operand,
            running_mean: Operand,
            running_variance: Operand,
            builder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.batch_norm_training(
                in0,
                scale,
                offset,
                running_mean,
                running_variance,
                epsilon=1e-5,
                dimension=1,
                momentum=0.1,
            )

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(isolation=True, accumulation=True),
    ) as report:
        compile_and_execute_ttir(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    bnt_numerics = [
        r
        for r in records
        if r.op == "ttnn.batch_norm_training" and r.check == "numerics"
    ]
    assert bnt_numerics, (
        "no numerics records produced for ttnn.batch_norm_training; "
        f"saw: {[(r.op, r.check) for r in records]}"
    )

    PCC_THRESHOLD = 0.99
    for mode in (chisel.NumericsMode.ISOLATED, chisel.NumericsMode.ACCUMULATED):
        for_mode = [r for r in bnt_numerics if r.payload.mode == mode]
        assert for_mode, f"no batch_norm_training numerics record for mode={mode}"
        # One record per SSA result + per in-place mutated operand. Distinct
        # SSAs prove the in-place running_mean / running_var were validated
        # alongside the SSA result.
        ssas = {r.ssa for r in for_mode}
        assert len(ssas) >= 2, (
            f"expected numerics records for the SSA result plus at least one "
            f"in-place operand on ttnn.batch_norm_training in {mode} mode, "
            f"got ssas={ssas}"
        )
        for r in for_mode:
            assert (
                r.status == chisel.RecordStatus.OK
            ), f"batch_norm_training {mode} fail: payload={r.payload}"
            assert (
                r.payload.pcc >= PCC_THRESHOLD
            ), f"batch_norm_training {mode} PCC {r.payload.pcc} below {PCC_THRESHOLD}"

    # When a chisel golden is registered, the pool entry for an in-place
    # operand is refreshed, not evicted.
    evictions = [
        r
        for r in records
        if r.op == "ttnn.batch_norm_training" and r.check == "golden_evicted"
    ]
    assert not evictions, (
        f"unexpected golden_evicted records on ttnn.batch_norm_training: "
        f"{[r.ssa for r in evictions]}"
    )
