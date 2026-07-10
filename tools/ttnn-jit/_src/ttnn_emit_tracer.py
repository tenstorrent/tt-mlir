# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Direct-TTNN emission tracer: monkeypatch ttnn.<op> to build TTNN IR directly.

The default advisor tracer (interception_tracer) emits TTIR and relies on the
compiler's ttnn-layout + convert-ttir-to-ttnn passes to reach TTNN. This variant
emits the TTNN dialect straight from the traced ops, with a default
DRAM-interleaved layout on every tensor, so the advisor can run the
`ttnn-to-ttnn-l1-advisor` pipeline (no lowering). An op then needs only a TTNN
op def -- no TTIR op + TTIRToTTNN conversion.

Layout synthesis is a single call reusing the C++ TTNNLayoutAttr::Builder
(default_ttnn_layout). This is 1:1 op->op like the TTIR tracer: no decomposition.

Coverage is the dense compute path (matmul/linear, elementwise, norm, reshape).
Attention / KV-cache / experimental ops are not yet ported to direct-TTNN; trace
those through the default TTIR path (pipeline="scoped").
"""

from contextlib import contextmanager

from ttnn_jit.ttmlir.dialects import func, ttnn, ttcore
from ttnn_jit.ttmlir.ir import (
    Location,
    InsertionPoint,
    RankedTensorType,
)

import ttnn as _ttnn_rt

from ttnn_jit._src.tracing_compiler import JitContext
from ttnn_jit._src.interception_tracer import (
    TracedTensor,
    TraceScope,
    _traced_element_type,
    _broadcast_batch,
    _restore_patched,
    _finalize_signature,
    _MISSING,
    Context,
    Module,
)

# BufferType.DRAM, TensorMemoryLayout.Interleaved (see project memory).
_DRAM = 0
_INTERLEAVED = 0


def _ttcore_dtype(elem_type):
    """ttcore.DataType for an MLIR scalar element type (incl. signed si32)."""
    s = str(elem_type)
    if s == "bf16":
        return ttcore.DataType.BFloat16
    if s == "f32":
        return ttcore.DataType.Float32
    if s in ("i32", "si32"):
        return ttcore.DataType.Int32
    if "bfp_bf8" in s.lower():
        return ttcore.DataType.BFP_BFloat8
    if "bfp_bf4" in s.lower():
        return ttcore.DataType.BFP_BFloat4
    raise ValueError(f"unsupported element type for TTNN layout: {elem_type}")


def default_ttnn_layout(ctx, shape, elem_type):
    """The whole 'layout synthesis': one call reusing TTNNLayoutAttr::Builder.

    Every direct-TTNN tensor gets a DRAM-interleaved default; the greedy
    optimizer reassigns L1/sharded layouts from there, exactly as it does for
    the layouts ttnn-layout assigns on the TTIR path.
    """
    tiled = ttcore.ir.TileType.get(ctx, 32, 32, _ttcore_dtype(elem_type))
    return ttnn.ir.TTNNLayoutAttr.get(
        ctx, list(shape), tiled, _DRAM, [1, 1], None, memLayout=_INTERLEAVED
    )


def _tt(ctx, shape, elem_type):
    """RankedTensorType carrying the default TTNN layout encoding."""
    shape = [int(d) for d in shape]
    return RankedTensorType.get(
        shape, elem_type, default_ttnn_layout(ctx, shape, elem_type)
    )


def _retype(ctx, value, shape, elem_type=None):
    """Convenience: build a layout'd result type from a value's element type."""
    et = elem_type if elem_type is not None else value.type.element_type
    return _tt(ctx, shape, et)


def build_ttnn_trace_scope(name, input_specs):
    """Module + func skeleton whose inputs carry the default TTNN layout."""
    ctx = Context()
    module = Module.create(Location.unknown(ctx))
    with Location.unknown(ctx):
        input_types = [
            _tt(ctx, shape, _traced_element_type(dtype, ctx))
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
    jit_ctx.cache_alias = {}
    traced_args = [TracedTensor(func_bb.arguments[i]) for i in range(len(input_types))]
    return TraceScope(ctx, module, func_op, func_bb, jit_ctx, traced_args, input_types)


def _weight_value(tensor, jit_ctx):
    """Materialize a captured weight as a layout'd placeholder, deduped by id.

    Uses a transient ttir.empty as the placeholder: _finalize_signature lifts
    every weight to a function argument (inheriting this layout'd type) and
    erases the placeholder, so no ttir op survives to the final TTNN module.
    """
    from ttnn_jit.ttmlir.dialects import ttir

    key = id(tensor)
    cache = jit_ctx.weight_cache
    if key in cache:
        return cache[key]
    shape = [int(d) for d in tensor.shape]
    elem = _traced_element_type(tensor.dtype, jit_ctx.ctx)
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        value = ttir.EmptyOp(_tt(jit_ctx.ctx, shape, elem)).result
    cache[key] = value
    return value


def _capture(arg, jit_ctx):
    if isinstance(arg, (list, tuple)):
        return type(arg)(_capture(a, jit_ctx) for a in arg)
    if type(arg) is TracedTensor or isinstance(arg, (int, float, bool)) or arg is None:
        return arg
    if hasattr(arg, "mlir_value"):
        return arg
    if hasattr(arg, "shape") and hasattr(arg, "dtype"):
        return TracedTensor(_weight_value(arg, jit_ctx))
    return arg


def _make_value_op(value_fn, jit_ctx):
    def op(*args, **kwargs):
        pa = [_capture(a, jit_ctx) for a in args]
        pk = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        return TracedTensor(value_fn(jit_ctx, *pa, **pk))

    return op


# ---------------------------------------------------------------------------
# Handlers -- each emits exactly one ttnn.<op> with a default-layout result.
# ---------------------------------------------------------------------------


def _matmul_out_shape(a, b, transpose_a, transpose_b):
    ash = [int(d) for d in a.shape]
    bsh = [int(d) for d in b.shape]
    if transpose_a:
        ash[-1], ash[-2] = ash[-2], ash[-1]
    if transpose_b:
        bsh[-1], bsh[-2] = bsh[-2], bsh[-1]
    batch = _broadcast_batch(ash[:-2], bsh[:-2])
    return batch + [ash[-2], bsh[-1]]


def _matmul_handler(jit_ctx, a, b, *, transpose_a=False, transpose_b=False, **kwargs):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        out = _matmul_out_shape(
            a.mlir_value.type, b.mlir_value.type, transpose_a, transpose_b
        )
        rt = _retype(jit_ctx.ctx, a.mlir_value, out)
        return ttnn.matmul(
            result=rt,
            a=a.mlir_value,
            b=b.mlir_value,
            transpose_a=bool(transpose_a),
            transpose_b=bool(transpose_b),
        )


def _linear_handler(
    jit_ctx, a, b, *, bias=None, transpose_a=False, transpose_b=False, **kwargs
):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        out = _matmul_out_shape(
            a.mlir_value.type, b.mlir_value.type, transpose_a, transpose_b
        )
        rt = _retype(jit_ctx.ctx, a.mlir_value, out)
        return ttnn.linear(
            result=rt,
            a=a.mlir_value,
            b=b.mlir_value,
            bias=(bias.mlir_value if bias is not None else None),
            transpose_a=bool(transpose_a),
            transpose_b=bool(transpose_b),
        )


def _binary(op_fn):
    def handler(jit_ctx, x, y, **kwargs):
        with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
            # scalar operand -> not supported in analysis; require two tensors.
            xs = [int(d) for d in x.mlir_value.type.shape]
            ys = [int(d) for d in y.mlir_value.type.shape]
            n = max(len(xs), len(ys))
            xr = [1] * (n - len(xs)) + xs
            yr = [1] * (n - len(ys)) + ys
            out = [a if b == 1 else b if a == 1 else max(a, b) for a, b in zip(xr, yr)]
            rt = _retype(jit_ctx.ctx, x.mlir_value, out)
            return op_fn(result=rt, lhs=x.mlir_value, rhs=y.mlir_value)

    return handler


def _unary(op_fn):
    def handler(jit_ctx, x, **kwargs):
        with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
            shape = [int(d) for d in x.mlir_value.type.shape]
            rt = _retype(jit_ctx.ctx, x.mlir_value, shape)
            return op_fn(result=rt, input=x.mlir_value)

    return handler


def _reshape_handler(jit_ctx, x, shape=None, **kwargs):
    dims = [int(d) for d in shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, dims)
        return ttnn.reshape(result=rt, input=x.mlir_value, shape=dims)


def _typecast_handler(jit_ctx, x, dtype, **kwargs):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        shape = [int(d) for d in x.mlir_value.type.shape]
        rt = _tt(jit_ctx.ctx, shape, _traced_element_type(dtype, jit_ctx.ctx))
        return ttnn.typecast(result=rt, input=x.mlir_value)


def _softmax_handler(jit_ctx, x, dim=None, **kwargs):
    d = int(dim) if dim is not None else -1
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        shape = [int(d) for d in x.mlir_value.type.shape]
        rt = _retype(jit_ctx.ctx, x.mlir_value, shape)
        return ttnn.softmax(result=rt, input=x.mlir_value, dimension=d)


def _rms_norm_handler(jit_ctx, x, *, epsilon=1e-5, weight=None, **kwargs):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        shape = [int(d) for d in x.mlir_value.type.shape]
        rt = _retype(jit_ctx.ctx, x.mlir_value, shape)
        return ttnn.rms_norm(
            result=rt,
            input=x.mlir_value,
            weight=(weight.mlir_value if weight is not None else None),
            epsilon=float(epsilon),
        )


_VALUE_HANDLERS = {
    "matmul": _matmul_handler,
    "linear": _linear_handler,
    "add": _binary(ttnn.add),
    "multiply": _binary(ttnn.multiply),
    "mul": _binary(ttnn.multiply),
    "subtract": _binary(ttnn.subtract),
    "reshape": _reshape_handler,
    "typecast": _typecast_handler,
    "softmax": _softmax_handler,
    "rms_norm": _rms_norm_handler,
    "relu": _unary(ttnn.relu),
    "gelu": _unary(ttnn.gelu),
    "silu": _unary(ttnn.silu),
    "sigmoid": _unary(ttnn.sigmoid),
    "sqrt": _unary(ttnn.sqrt),
    "rsqrt": _unary(ttnn.rsqrt),
    "exp": _unary(ttnn.exp),
    "neg": _unary(ttnn.neg),
    "tanh": _unary(ttnn.tanh),
    "reciprocal": _unary(ttnn.reciprocal),
    "abs": _unary(ttnn.abs),
}

# Layout ops are no-ops for analysis: return the tensor unchanged (see the TTIR
# tracer's _PASSTHROUGH sets). The optimizer inserts real reshards itself.
_PASSTHROUGH_IDENTITY = {
    "to_memory_config",
    "to_layout",
    "interleaved_to_sharded",
    "sharded_to_interleaved",
}
_PASSTHROUGH_NONE = {"deallocate"}


def _identity_passthrough(*args, **kwargs):
    for a in args:
        if type(a) is TracedTensor:
            return a
    for v in kwargs.values():
        if type(v) is TracedTensor:
            return v
    return args[0] if args else None


def _none_passthrough(*args, **kwargs):
    return None


@contextmanager
def patch_ttnn(jit_ctx):
    """Monkeypatch allowlisted ttnn.<op> to build TTNN directly; restore on exit."""
    originals = {}
    try:
        for name in _PASSTHROUGH_IDENTITY:
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _identity_passthrough)
        for name in _PASSTHROUGH_NONE:
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _none_passthrough)
        for name, value_fn in _VALUE_HANDLERS.items():
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_value_op(value_fn, jit_ctx))
        yield
    finally:
        _restore_patched(_ttnn_rt, originals)


def trace_ttnn(fn, *args):
    """Trace `fn` (called with example tensors) to a TTNN module directly.

    `args` are example tensors (only .shape/.dtype are read). Returns (module,
    output_type). Feed the result to the `ttnn-to-ttnn-l1-advisor` pipeline.
    """
    input_specs = [(tuple(int(d) for d in a.shape), a.dtype) for a in args]
    scope = build_ttnn_trace_scope(fn.__name__, input_specs)

    with patch_ttnn(scope.jit_ctx):
        result = fn(*scope.traced_args)

    if type(result) is not TracedTensor:
        raise TypeError(
            f"traced function must return a single tensor, got {type(result)!r}"
        )

    return_value = result.mlir_value
    weight_values = list(scope.jit_ctx.weight_cache.values())
    _finalize_signature(
        scope.module,
        scope.func_op,
        scope.input_types,
        return_value,
        scope.ctx,
        weight_values,
    )
    scope.module.operation.verify()
    return scope.module, return_value.type
