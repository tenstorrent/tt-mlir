# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-program golden accumulation: outputs of one program flow into the
next program's inputs and seed chisel's golden chain from the cross-program
pool (source="session_pool"), rather than re-seeding from the device.

Two scenarios:
  * A->B: two distinct compiled programs, the output of A becomes input to B.
  * A->A->A: the same program executed three times, each run consumes the
    previous run's output.
"""
import os
from typing import List, Optional

import torch
import _ttmlir_runtime as tt_runtime

import chisel
from builder.base.builder_apis import compile_ttnn_to_flatbuffer
from builder.base.builder_runtime import (
    convert_input_layouts,
    create_tensor,
)
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder


_SHAPE = (32, 32)


def _compile(builder_fn, test_base: str, tmp_path) -> tt_runtime.binary.Binary:
    artifact_dir = str(tmp_path / test_base)
    system_desc_path = os.environ.get(
        "SYSTEM_DESC_PATH", "ttrt-artifacts/system_desc.ttsys"
    )
    _, capsule, _, _ = compile_ttnn_to_flatbuffer(
        builder_fn,
        artifact_dir=artifact_dir,
        target="ttnn",
        system_desc_path=system_desc_path,
    )
    return tt_runtime.binary.load_binary_from_capsule(capsule)


def _make_host_input(device, shape=_SHAPE, dtype=torch.float32):
    """Build a random borrowed host tensor on the device's mesh shape."""
    torch_t = torch.randn(shape, dtype=dtype)
    return create_tensor({0: torch_t}, device.get_mesh_shape())


def _submit_with_layout_convert(device, fbb, host_inputs, program_index: int = 0):
    """First submit of a binary: convert all host inputs to the binary's
    expected layouts, then submit."""
    converted = convert_input_layouts(
        device, host_inputs, fbb=fbb, program_index=program_index
    )
    outputs = tt_runtime.runtime.submit(device, fbb, program_index, converted)
    tt_runtime.runtime.wait(outputs)
    return outputs


def _submit_chained(device, fbb, inputs, program_index: int = 0):
    """Submit where some inputs are device tensors carried over from a prior
    program's output. Those tensors are passed through verbatim so their
    runtime Tensor identity (and globalId) survives - that is what lets
    chisel match them to its cross-program pool. Caller is responsible for
    ensuring the layouts are compatible."""
    outputs = tt_runtime.runtime.submit(device, fbb, program_index, list(inputs))
    tt_runtime.runtime.wait(outputs)
    return outputs


def _session_pool_records(records):
    return [
        r
        for r in records
        if r.check == "golden_promoted"
        and r.payload.source == chisel.GoldenPromotionSource.SESSION_POOL
    ]


def _device_seeded_records(records):
    return [
        r
        for r in records
        if r.check == "golden_promoted"
        and r.payload.source == chisel.GoldenPromotionSource.DEVICE
    ]


def test_session_pool_chains_a_to_b(device, tmp_path):
    """Output of program A is fed as input to program B; chisel should
    record at least one SESSION_POOL promotion on B's first op."""

    def prog_a(builder: TTNNBuilder):
        @builder.func([_SHAPE, _SHAPE], [torch.float32, torch.float32])
        def prog_a_fn(
            x: Operand,
            y: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.multiply(x, y)

    def prog_b(builder: TTNNBuilder):
        @builder.func([_SHAPE, _SHAPE], [torch.float32, torch.float32])
        def prog_b_fn(
            u: Operand,
            v: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.add(u, v)

    fbb_a = _compile(prog_a, "prog_a", tmp_path)
    fbb_b = _compile(prog_b, "prog_b", tmp_path)

    with chisel.session(
        checks_config=chisel.ChiselChecksConfig(accumulation=True),
    ) as report:
        x_host = _make_host_input(device)
        y_host = _make_host_input(device)
        out_a = _submit_with_layout_convert(device, fbb_a, [x_host, y_host])
        assert out_a, "program A produced no outputs"

        # While A's output is still live, the cross-program pool must hold
        # the published golden.
        ctx = chisel.context.get_instance()
        assert (
            ctx.program_io_pool
        ), "program_io_pool should contain A's published golden between submits"

        # Convert B's second (fresh) input via the binary's expected layout
        # for slot 1. Slot 0 is A's device output, passed through unchanged
        # to preserve runtime Tensor identity for the cross-program lookup.
        v_host = _make_host_input(device)
        v_layout = tt_runtime.runtime.get_layout(fbb_b, 0, 1)
        v_dev = tt_runtime.runtime.to_layout(v_host, device, v_layout, True)

        out_b = _submit_chained(device, fbb_b, [out_a[0], v_dev])
        assert out_b, "program B produced no outputs"

        records = list(report.records)

    pool_records = _session_pool_records(records)
    assert pool_records, (
        "expected at least one golden_promoted record with source='session_pool' "
        "on program B's first op; promotions seen: "
        f"{[(r.op, r.ssa, r.payload.source) for r in records if r.check == 'golden_promoted']}"
    )


def test_session_pool_chains_a_to_a_to_a(device, tmp_path):
    """Re-running the same program three times, each run consuming the prior
    run's output. Runs 2 and 3 should each see at least one SESSION_POOL
    promotion on their first op."""

    def prog(builder: TTNNBuilder):
        @builder.func([_SHAPE, _SHAPE], [torch.float32, torch.float32])
        def prog_fn(
            x: Operand,
            y: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.multiply(x, y)

    fbb = _compile(prog, "prog_a_a_a", tmp_path)

    with chisel.session(
        checks_config=chisel.ChiselChecksConfig(accumulation=True),
    ) as report:
        x_host = _make_host_input(device)
        y_host = _make_host_input(device)

        out1 = _submit_with_layout_convert(device, fbb, [x_host, y_host])
        ctx = chisel.context.get_instance()
        pool_size_after_run1 = len(ctx.program_io_pool)
        assert pool_size_after_run1 >= 1, (
            "expected at least one entry in program_io_pool after run 1, "
            f"got {pool_size_after_run1}"
        )

        # Pre-layout y for the subsequent submits so it matches what slot 1
        # expects; slot 0 takes the prior run's device output as-is.
        y_layout = tt_runtime.runtime.get_layout(fbb, 0, 1)
        y_dev = tt_runtime.runtime.to_layout(y_host, device, y_layout, True)

        out2 = _submit_chained(device, fbb, [out1[0], y_dev])
        out3 = _submit_chained(device, fbb, [out2[0], y_dev])
        assert out3, "third submit produced no outputs"

        records = list(report.records)

    # Runs 2 and 3 should each contribute at least one session_pool promotion
    # on the first op (the chained input arg).
    pool_records = _session_pool_records(records)
    assert len(pool_records) >= 2, (
        f"expected at least 2 session_pool promotions across 3 runs, got "
        f"{len(pool_records)}: {[(r.op, r.ssa) for r in pool_records]}"
    )

    # Sanity: device-seeded promotions still happen for run 1 and for the
    # never-chained `y` input across runs.
    device_records = _device_seeded_records(records)
    assert device_records, (
        "expected at least one device-seeded golden_promoted record "
        "(e.g. for fresh inputs of the first run)"
    )
