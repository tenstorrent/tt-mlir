# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device tests for skip mode.

Exercises the full skip path on silicon: predicate -> isolation golden ->
`update_tensor_in_pool` write-back. The `device` fixture (conftest) opens a
single-chip mesh.
"""
from collections import OrderedDict
from typing import List, Optional, Sequence

import torch

import chisel
from chisel.callbacks import _default_pre_op
from ttmlir.dialects import ttnn
from builder.base.builder_apis import compile_and_execute_ttnn
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder

from golden import CHISEL_GOLDEN_MAPPINGS, GoldenMapTensor


# n300-style (1, 2) mesh; mirrors test_builder_chisel_integration.py. The
# multichip_device fixture (conftest) opens a matching mesh and skips cleanly on
# hosts with fewer chips.
_MULTICHIP_MESH = OrderedDict([("x", 1), ("y", 2)])
_STYLE_A_PIPELINE = (
    "ttcore-mark-functions-as-forward,"
    "ttcore-wrap-device-module,"
    "ttcore.device_module(builtin.module("
    "ttnn-configure-ccl-ops,ttnn-deallocate))"
)
_SYSTEM_MEM_RM = {
    "layout": ttnn.Layout.RowMajor,
    "buffer_type": ttnn.BufferType.SystemMemory,
}


# linear: x(64,128) @ w(256,128)^T + b(256) -> (64,256); multiply(y, y) stays a
# separate op (it is not an activation, so the optimizer does not fuse it).
_X_SHAPE = (64, 128)
_W_SHAPE = (256, 128)
_B_SHAPE = (256,)
_Y_SHAPE = (64, 256)


def _one_layer(builder: TTNNBuilder):
    @builder.func(
        [_X_SHAPE, _W_SHAPE, _B_SHAPE],
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

    return one_layer_nn


def test_skip_op_emits_record_and_completes(request, device, tmp_path):
    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(
            accumulation=True,
            skip_op=chisel.skip.skip_op_names("ttnn.multiply"),
        ),
    ) as report:
        compile_and_execute_ttnn(
            _one_layer,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    skipped = [r for r in records if r.check == "skipped_op"]
    assert skipped, "no skipped_op record produced"
    assert {r.op for r in skipped} == {"ttnn.multiply"}
    assert all(r.status is chisel.RecordStatus.SKIPPED_OP for r in skipped)

    # A skip is an action, not a chisel failure - the run must stay clean.
    bugs = [r for r in records if r.status is chisel.RecordStatus.CHISEL_BUG]
    assert not bugs, f"chisel_bug records present: {bugs}"


def test_skip_op_substitutes_device_output(request, device, tmp_path, monkeypatch):
    """Hard-check that the device tensor is actually overwritten.

    Force ttnn.linear's golden to a known constant (all 2s) and skip it. The
    skip writes that golden over linear's device output, so the downstream
    multiply must see all-2s inputs. We capture multiply's stashed device inputs
    via a custom pre_op to assert the substitution reached the device.
    """
    const = 2.0

    def _fake_linear_golden(op, inputs):
        return GoldenMapTensor({0: torch.full(_Y_SHAPE, const)}, (1, 1))

    monkeypatch.setitem(CHISEL_GOLDEN_MAPPINGS, ttnn.LinearOp, _fake_linear_golden)

    captured: List[torch.Tensor] = []

    def _capturing_pre_op(ctx, config):
        # Delegate to the default handler (stashes device inputs into the pool),
        # then snapshot multiply's device inputs - i.e. linear's (now skipped)
        # device output.
        ok = _default_pre_op(ctx, config)
        for golden in ctx.stashed_inputs.values():
            captured.append(golden.shard_map[0].clone())
        return ok

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(
            accumulation=True,
            skip_op=chisel.skip.skip_op_names("ttnn.linear"),
        ),
    ) as report:
        chisel.register_op_config(
            ttnn.MultiplyOp, chisel.ChiselOpConfig(pre_op=_capturing_pre_op)
        )
        compile_and_execute_ttnn(
            _one_layer,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert any(
        r.check == "skipped_op" and r.op == "ttnn.linear" for r in records
    ), "ttnn.linear was not skipped"
    assert captured, "multiply pre_op captured no inputs"
    # multiply(y, y) reads the skipped linear output; it must now be all-2s.
    for tensor in captured:
        assert torch.allclose(
            tensor, torch.full_like(tensor, const)
        ), f"device input not substituted: mean={tensor.mean().item()}"


# ---------------------------------------------------------------------------
# Multichip: skip the matmul on an n300-style (1, 2) mesh. Exercises the
# multi-device write-back path (golden_to_runtime_tensor -> from_host_shards ->
# update_tensor_in_pool with one shard per chip). Mirrors the matmul layout of
# test_builder_chisel_integration.py's multichip tests.
# ---------------------------------------------------------------------------


def _build_matmul_module(
    builder: TTNNBuilder,
    x_shape: Sequence[int],
    w_shape: Sequence[int],
    x_shard_dims: List[int],
    w_shard_dims: List[int],
    out_shard_dims: List[int],
):
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


def _run_skip_matmul_multichip(request, device, tmp_path, x_shard, w_shard, out_shard):
    x_shape = (64, 128)
    w_shape = (128, 256)

    def module(builder: TTNNBuilder):
        _build_matmul_module(
            builder,
            x_shape=x_shape,
            w_shape=w_shape,
            x_shard_dims=x_shard,
            w_shard_dims=w_shard,
            out_shard_dims=out_shard,
        )

    with chisel.session(
        results_path=str(tmp_path / "chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(
            isolation=True,
            accumulation=True,
            skip_op=chisel.skip.skip_op_names("ttnn.matmul"),
        ),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
            mesh_dict=_MULTICHIP_MESH,
            custom_pipeline=_STYLE_A_PIPELINE,
        )
        records = report.records
    return records


def _assert_skipped_matmul_multichip(records):
    assert records, "chisel report is empty"
    # Per-chip PCC still recorded (skip runs after the checks): two device_ids.
    matmul_pcc = [r for r in records if r.op == "ttnn.matmul" and r.check == "numerics"]
    assert matmul_pcc, "no ttnn.matmul PCC records"
    assert {r.payload.device_id for r in matmul_pcc} == {0, 1}, (
        "expected per-chip matmul PCC records for device_ids {0, 1}, got "
        f"{sorted({r.payload.device_id for r in matmul_pcc})}"
    )
    # The multi-device write-back succeeded: one skipped_op record, no bugs.
    skipped = [r for r in records if r.check == "skipped_op" and r.op == "ttnn.matmul"]
    assert skipped, "ttnn.matmul was not skipped"
    bugs = [r for r in records if r.status is chisel.RecordStatus.CHISEL_BUG]
    assert not bugs, f"chisel_bug records present (write-back failed?): {bugs}"


def test_skip_matmul_data_parallel_multichip(request, multichip_device, tmp_path):
    # x split along batch (dim 0) across mesh axis 1, w replicated.
    records = _run_skip_matmul_multichip(
        request,
        multichip_device,
        tmp_path,
        x_shard=[-1, 0],
        w_shard=[-1, -1],
        out_shard=[-1, 0],
    )
    _assert_skipped_matmul_multichip(records)


def test_skip_matmul_tensor_parallel_multichip(request, multichip_device, tmp_path):
    # x replicated, w split along output dim (dim 1) across mesh axis 1.
    records = _run_skip_matmul_multichip(
        request,
        multichip_device,
        tmp_path,
        x_shard=[-1, -1],
        w_shard=[-1, 1],
        out_shard=[-1, 1],
    )
    _assert_skipped_matmul_multichip(records)
