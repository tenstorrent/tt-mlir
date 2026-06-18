# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cross-program golden chaining through sub-program ops (ttcore.load_cached).

The op delegates execution to a nested program. Chisel reuses the
session-scoped globalId pool: the parent op's pre-handler publishes each
input's accumulated golden keyed by the input Tensor's globalId, the
sub-program's default pre-op finds it as a function-arg cross-pool hit, the
sub-program's default post-op republishes its function-output goldens by
globalId, and the parent op's post-handler reads them back to install the
golden on the parent SSA. See tools/chisel/chisel/op_handlers.py:
_subprogram_pre_op / _subprogram_post_op.

Two scenarios:
  * ttcore.load_cached, cache miss: parameter-typed args trigger const-eval
    hoisting; first submit runs the const-eval sub-function.
  * ttcore.load_cached, cache hit (same session): second submit with the same
    device-input wrappers should hit the cache and still find the chained
    golden via the surviving pool entries.
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


def _compile(
    builder_fn,
    test_base: str,
    tmp_path,
    *,
    argument_types_string: Optional[str] = None,
) -> tt_runtime.binary.Binary:
    artifact_dir = str(tmp_path / test_base)
    system_desc_path = os.environ.get(
        "SYSTEM_DESC_PATH", "ttrt-artifacts/system_desc.ttsys"
    )
    _, capsule, _, _ = compile_ttnn_to_flatbuffer(
        builder_fn,
        artifact_dir=artifact_dir,
        target="ttnn",
        system_desc_path=system_desc_path,
        argument_types_string=argument_types_string,
    )
    return tt_runtime.binary.load_binary_from_capsule(capsule)


def _make_host_input(device, shape=_SHAPE, dtype=torch.float32):
    torch_t = torch.randn(shape, dtype=dtype)
    return create_tensor({0: torch_t}, device.get_mesh_shape())


def _submit_with_layout_convert(device, fbb, host_inputs, program_index: int = 0):
    converted = convert_input_layouts(
        device, host_inputs, fbb=fbb, program_index=program_index
    )
    outputs = tt_runtime.runtime.submit(device, fbb, program_index, converted)
    tt_runtime.runtime.wait(outputs)
    return outputs, converted


def _submit_passthrough(device, fbb, device_inputs, program_index: int = 0):
    """Submit with already-laid-out device tensors. Reusing the same Tensor
    objects keeps their TTNNTensorWrapper versions stable - the input to
    LoadCachedOp's cache-key construction. Same versions => cache hit."""
    outputs = tt_runtime.runtime.submit(device, fbb, program_index, list(device_inputs))
    tt_runtime.runtime.wait(outputs)
    return outputs


def _session_pool_records(records):
    return [
        r
        for r in records
        if r.check == "golden_promoted"
        and r.payload.source == chisel.GoldenPromotionSource.SESSION_POOL
    ]


def test_cache_miss_chains_through_subprogram(device, tmp_path):
    """argument_types_string tags two args as parameters; the const-eval hoist
    pass moves the parameter-only subgraph into a ttcore.load_cached call.
    First submit is a cache miss: the sub-function runs and its goldens flow
    back into the parent via the cross-program pool."""

    def prog(builder: TTNNBuilder):
        @builder.func(
            [_SHAPE, _SHAPE, _SHAPE],
            [torch.float32, torch.float32, torch.float32],
        )
        def forward(
            x: Operand,
            p1: Operand,
            p2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # multiply(p1, p2) depends only on parameters - hoistable into
            # a const-eval sub-function; the resulting LoadCachedOp output is
            # the value chisel must chain through.
            prod = builder.multiply(p1, p2)
            return builder.add(x, prod)

    fbb = _compile(
        prog,
        "load_cached_miss_prog",
        tmp_path,
        argument_types_string="forward=input,parameter,parameter",
    )
    assert fbb.get_num_programs() >= 2, (
        "const-eval hoist did not produce a sub-program; argument_types_string "
        "may have been ignored. num_programs="
        f"{fbb.get_num_programs()}, names="
        f"{[fbb.get_program_name(i) for i in range(fbb.get_num_programs())]}"
    )

    parent_name = fbb.get_program_name(0)

    with chisel.session(
        checks_config=chisel.ChiselChecksConfig(accumulation=True),
    ) as report:
        x_host = _make_host_input(device)
        p1_host = _make_host_input(device)
        p2_host = _make_host_input(device)
        outs, _ = _submit_with_layout_convert(device, fbb, [x_host, p1_host, p2_host])
        assert outs, "load_cached program produced no outputs"
        records = list(report.records)

    pool_records = _session_pool_records(records)
    assert pool_records, (
        "expected source=session_pool records from the load_cached chain; "
        f"promotions seen: "
        f"{[(r.op, r.ssa, r.payload.source, r.program_name) for r in records if r.check == 'golden_promoted']}"
    )

    # Parent's LoadCachedOp output is promoted via the program pool.
    parent_loadcached_pool = [
        r
        for r in pool_records
        if r.program_name == parent_name and r.op == "ttcore.load_cached"
    ]
    assert parent_loadcached_pool, (
        "expected source=session_pool record on the parent's ttcore.load_cached "
        f"output; pool records: "
        f"{[(r.op, r.program_name) for r in pool_records]}"
    )


def test_cache_hit_chains_through_subprogram(device, tmp_path):
    """Submit the same program twice in one session, reusing the same device
    tensor wrappers as inputs on the second submit so wrapper versions are
    unchanged and the LoadCachedOp hits its cache. On the cache hit the
    sub-function does NOT run, but the cached output Tensors carry the
    original globalIds and their entries are still in the cross-program pool
    (retained tensors, no destroy callback fired). The parent's load_cached
    post-op must still see source=session_pool on the second run."""

    def prog(builder: TTNNBuilder):
        @builder.func(
            [_SHAPE, _SHAPE, _SHAPE],
            [torch.float32, torch.float32, torch.float32],
        )
        def forward(
            x: Operand,
            p1: Operand,
            p2: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            prod = builder.multiply(p1, p2)
            return builder.add(x, prod)

    fbb = _compile(
        prog,
        "load_cached_hit_prog",
        tmp_path,
        argument_types_string="forward=input,parameter,parameter",
    )
    assert fbb.get_num_programs() >= 2

    parent_name = fbb.get_program_name(0)

    with chisel.session(
        checks_config=chisel.ChiselChecksConfig(accumulation=True),
    ) as report:
        x_host = _make_host_input(device)
        p1_host = _make_host_input(device)
        p2_host = _make_host_input(device)
        out1, dev_inputs = _submit_with_layout_convert(
            device, fbb, [x_host, p1_host, p2_host]
        )
        assert out1, "first submit produced no outputs"
        records_run1 = list(report.records)

        # Pool must contain at least the LoadCachedOp output's golden between
        # submits. Without a surviving entry the cache hit cannot chain.
        ctx = chisel.context.get_instance()
        assert (
            ctx.session_pool
        ), "session_pool empty between submits; cache hit cannot chain"

        # Second submit: pass the same device tensors verbatim. Same
        # TTNNTensorWrappers => same versions => LoadCachedOp cache hit.
        out2 = _submit_passthrough(device, fbb, dev_inputs)
        assert out2, "second submit produced no outputs"
        records_all = list(report.records)

    records_run2 = records_all[len(records_run1) :]

    run2_pool_records = _session_pool_records(records_run2)
    assert run2_pool_records, (
        "expected source=session_pool records on run 2 (cache hit); "
        f"run-2 promotions: "
        f"{[(r.op, r.ssa, r.payload.source, r.program_name) for r in records_run2 if r.check == 'golden_promoted']}"
    )

    run2_loadcached_pool = [
        r
        for r in run2_pool_records
        if r.program_name == parent_name and r.op == "ttcore.load_cached"
    ]
    assert run2_loadcached_pool, (
        "run 2 (cache hit): expected source=session_pool on the parent's "
        "ttcore.load_cached output; pool records were "
        f"{[(r.op, r.program_name) for r in run2_pool_records]}"
    )
