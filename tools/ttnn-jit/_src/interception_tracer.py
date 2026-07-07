# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Interception tracer: monkeypatch ttnn.<op> to build TTIR across call boundaries.

Reuses the existing BaseOpHandler machinery (jit_functions.py). Only dispatch,
the proxy tensor, and weight capture differ from the source-rewrite tracer.
"""

from ttnn_jit.ttmlir.dialects import func, ttir
from ttnn_jit.ttmlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
)

from ttnn_jit._src.tracing_compiler import JitContext
from ttnn_jit._src.conversions import (
    mlir_dtype_from_ttnn_dtype,
    ttnn_dtype_from_mlir_dtype,
)


class TracedTensor:
    """Proxy standing in for a tensor during tracing.

    Holds the MLIR Value for the tensor and exposes shape/dtype (read from the
    MLIR result type) so the traced model's own Python shape math works. Layout
    is intentionally unknown -- the optimizer assigns it.
    """

    def __init__(self, mlir_value):
        self.mlir_value = mlir_value

    @property
    def shape(self):
        return tuple(int(d) for d in self.mlir_value.type.shape)

    @property
    def dtype(self):
        return ttnn_dtype_from_mlir_dtype(self.mlir_value.type.element_type)

    def memory_config(self):
        raise NotImplementedError(
            "memory_config is unknown during analysis; the optimizer assigns it"
        )

    @property
    def layout(self):
        raise NotImplementedError(
            "layout is unknown during analysis; the optimizer assigns it"
        )


class TraceScope:
    """Bundle of MLIR objects for one trace."""

    def __init__(
        self, ctx, module, func_op, func_bb, jit_ctx, traced_args, input_types
    ):
        self.ctx = ctx
        self.module = module
        self.func_op = func_op
        self.func_bb = func_bb
        self.jit_ctx = jit_ctx
        self.traced_args = traced_args
        self.input_types = input_types


def build_trace_scope(name, input_specs):
    """Create an MLIR module + func skeleton with the given (shape, dtype) inputs.

    Inputs carry NO layout encoding (plain RankedTensorType) -- the compiler's
    TTNNLayout pass assigns layouts downstream, keeping this analysis-agnostic.
    """
    ctx = Context()
    module = Module.create(Location.unknown(ctx))
    with Location.unknown(ctx):
        input_types = [
            RankedTensorType.get(list(shape), mlir_dtype_from_ttnn_dtype(dtype, ctx))
            for shape, dtype in input_specs
        ]
        with InsertionPoint(module.body):
            func_op = func.FuncOp(
                name=name,
                type=(input_types, [input_types[0]] if input_types else []),
            )
            func_bb = func_op.add_entry_block()

    jit_ctx = JitContext(func_bb, ctx, (1, 1), (7, 7))
    jit_ctx.weight_cache = {}
    traced_args = [TracedTensor(func_bb.arguments[i]) for i in range(len(input_types))]
    return TraceScope(ctx, module, func_op, func_bb, jit_ctx, traced_args, input_types)
