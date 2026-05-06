# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: build a one-layer NN (linear + multiply) via TTNNBuilder, run with
chisel attached, and assert the in-memory report contains records for every op
we authored.
"""
from typing import List, Optional

import torch

import chisel
from builder.base.builder_apis import compile_and_execute_ttnn
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder


def test_chisel_records_one_layer_nn(request, device, tmp_path):
    """linear(x, W, b) → multiply(y, y) — assert both ops appear in chisel report.

    Note: sigmoid is NOT used here because the TTNN optimizer fuses it into
    ttnn.linear as an activation attribute, producing a single compiled op.
    multiply(y, y) cannot be fused and always appears as a distinct op.
    """
    x_shape = (64, 128)
    w_shape = (256, 128)  # transpose_b=True → effective rhs is (128, 256)
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

    chisel.bind()
    chisel.configure(
        results_path=None,
        isolation_check=False,
        strict=False,
    )
    try:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = chisel.get_report().records
    finally:
        chisel.unbind()

    assert records, "chisel report is empty"

    ops_seen = {r.op for r in records}
    expected = {"ttnn.linear", "ttnn.multiply"}
    missing = expected - ops_seen
    assert not missing, (
        f"chisel report missing op records for: {sorted(missing)}. "
        f"Saw: {sorted(ops_seen)}"
    )
