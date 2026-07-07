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
    F32Type,
    FloatAttr,
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


from contextlib import contextmanager

from ttnn_jit._src.jit_functions import TTNNJitNamespaceUpdater
from ttnn_jit._src.supported_ops import (
    unary_ops,
    binary_ops,
    reduction_ops,
    tm_ops,
    data_movement_ops,
)

_MISSING = object()

_PASSTHROUGH_IDENTITY = {
    "to_memory_config",
    "interleaved_to_sharded",
    "sharded_to_interleaved",
}
_PASSTHROUGH_NONE = {"deallocate"}


def _identity_passthrough(*args, **kwargs):
    # Layout op: return the first tensor operand unchanged; emit no TTIR.
    for a in args:
        if isinstance(a, TracedTensor):
            return a
    for v in kwargs.values():
        if isinstance(v, TracedTensor):
            return v
    return args[0] if args else None


def _none_passthrough(*args, **kwargs):
    return None


def _allowlisted_op_names():
    return (
        set(unary_ops)
        | set(binary_ops)
        | set(reduction_ops)
        | set(tm_ops)
        | set(data_movement_ops)
        | {"matmul", "div", "pow"}
    )


def _weight_value(tensor, jit_ctx):
    """Materialize a captured (non-proxy) tensor as a ttir.empty, deduped by id."""
    key = id(tensor)
    cache = jit_ctx.weight_cache
    if key in cache:
        return cache[key]
    shape = [int(d) for d in tensor.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        ttype = RankedTensorType.get(
            shape, mlir_dtype_from_ttnn_dtype(tensor.dtype, jit_ctx.ctx)
        )
        value = ttir.EmptyOp(ttype).result
    cache[key] = value
    return value


def _capture(arg, jit_ctx):
    if isinstance(arg, (list, tuple)):
        return type(arg)(_capture(a, jit_ctx) for a in arg)
    if (
        isinstance(arg, TracedTensor)
        or isinstance(arg, (int, float, bool))
        or arg is None
    ):
        return arg
    if hasattr(arg, "mlir_value"):
        return arg
    # A real tensor (weight/cache/table) -> materialize as a graph input.
    if hasattr(arg, "shape") and hasattr(arg, "dtype"):
        return TracedTensor(_weight_value(arg, jit_ctx))
    # Config/enum/other non-tensor object -> pass through unchanged.
    return arg


def _make_traced_op(handler_fn, jit_ctx):
    def op(*args, **kwargs):
        proxied_args = [_capture(a, jit_ctx) for a in args]
        proxied_kwargs = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        result = handler_fn(*proxied_args, **proxied_kwargs)
        return TracedTensor(result.mlir_value)

    return op


def _make_traced_value_op(value_fn, jit_ctx):
    def op(*args, **kwargs):
        proxied_args = [_capture(a, jit_ctx) for a in args]
        proxied_kwargs = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        return TracedTensor(value_fn(jit_ctx, *proxied_args, **proxied_kwargs))

    return op


def _wrap_results(results):
    """Wrap an MLIR OpResultList / list of values into a tuple of TracedTensors."""
    return tuple(TracedTensor(v) for v in results)


def _make_traced_multi_op(value_fn, jit_ctx):
    def op(*args, **kwargs):
        proxied_args = [_capture(a, jit_ctx) for a in args]
        proxied_kwargs = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        return _wrap_results(value_fn(jit_ctx, *proxied_args, **proxied_kwargs))

    return op


def _linear_handler(jit_ctx, a, b, *, bias=None, dtype=None, **kwargs):
    a_type = a.mlir_value.type
    b_type = b.mlir_value.type
    out_shape = [int(d) for d in a_type.shape[:-1]] + [int(b_type.shape[-1])]
    elem_type = (
        mlir_dtype_from_ttnn_dtype(dtype, jit_ctx.ctx)
        if dtype is not None
        else a_type.element_type
    )
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get(out_shape, elem_type)
        return ttir.linear(
            result=result_type,
            a=a.mlir_value,
            b=b.mlir_value,
            bias=(bias.mlir_value if bias is not None else None),
        )


def _typecast_handler(jit_ctx, x, dtype, **kwargs):
    x_type = x.mlir_value.type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get(
            [int(d) for d in x_type.shape],
            mlir_dtype_from_ttnn_dtype(dtype, jit_ctx.ctx),
        )
        return ttir.typecast(result=result_type, input=x.mlir_value)


def _rms_norm_handler(jit_ctx, x, *, epsilon=1e-5, weight=None, **kwargs):
    x_type = x.mlir_value.type
    hidden = int(x_type.shape[-1])
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get(
            [int(d) for d in x_type.shape], x_type.element_type
        )
        eps_attr = FloatAttr.get(F32Type.get(jit_ctx.ctx), float(epsilon))
        return ttir.rms_norm(
            result=result_type,
            input=x.mlir_value,
            normalized_shape=[hidden],
            weight=(weight.mlir_value if weight is not None else None),
            epsilon=eps_attr,
        )


_VALUE_HANDLERS = {
    "linear": _linear_handler,
    "typecast": _typecast_handler,
    "rms_norm": _rms_norm_handler,
}


@contextmanager
def patch_ttnn(jit_ctx):
    """Monkeypatch allowlisted ttnn.<op> to build TTIR; restore on exit."""
    import ttnn

    namespace = TTNNJitNamespaceUpdater(jit_ctx)
    originals = {}
    try:
        for name in _PASSTHROUGH_IDENTITY:
            originals[name] = getattr(ttnn, name, _MISSING)
            setattr(ttnn, name, _identity_passthrough)
        for name in _PASSTHROUGH_NONE:
            originals[name] = getattr(ttnn, name, _MISSING)
            setattr(ttnn, name, _none_passthrough)
        for name in _allowlisted_op_names():
            if not hasattr(namespace, name):
                continue
            handler_fn = getattr(namespace, name)
            originals[name] = getattr(ttnn, name, _MISSING)
            setattr(ttnn, name, _make_traced_op(handler_fn, jit_ctx))
        for name, value_fn in _VALUE_HANDLERS.items():
            originals[name] = getattr(ttnn, name, _MISSING)
            setattr(ttnn, name, _make_traced_value_op(value_fn, jit_ctx))
        if hasattr(namespace, "multiply"):
            originals["mul"] = getattr(ttnn, "mul", _MISSING)
            setattr(ttnn, "mul", _make_traced_op(getattr(namespace, "multiply"), jit_ctx))
        yield
    finally:
        for name, original in originals.items():
            if original is _MISSING:
                try:
                    delattr(ttnn, name)
                except AttributeError:
                    pass
            else:
                setattr(ttnn, name, original)


def _finalize_signature(module, func_op, input_types, return_value, ctx):
    """Add the return op and rebuild the func with the correct result type.

    Mirrors TracingCompiler._update_function_signature: MLIR can't mutate a func
    signature in place, so recreate the func and move the body block.
    """
    old_block = func_op.regions[0].blocks[0]
    if not old_block.operations or not isinstance(
        old_block.operations[-1], func.ReturnOp
    ):
        with InsertionPoint(old_block), Location.unknown(ctx):
            func.ReturnOp([return_value])

    with InsertionPoint(module.body), Location.unknown(ctx):
        new_func = func.FuncOp(
            name=func_op.name.value, type=(input_types, [return_value.type])
        )
    old_block.append_to(new_func.regions[0])
    func_op.erase()


def trace_intercepted(fn, *args):
    """Trace `fn` (called with example tensors) to a TTIR module via interception.

    `args` are example tensors (or metadata stand-ins) for `fn`'s declared
    parameters; only their `.shape`/`.dtype` are read. Returns (module,
    output_type). Output layout is left unforced so the optimizer chooses it.
    """
    input_specs = [(tuple(int(d) for d in a.shape), a.dtype) for a in args]
    scope = build_trace_scope(fn.__name__, input_specs)

    with patch_ttnn(scope.jit_ctx):
        result = fn(*scope.traced_args)

    if not isinstance(result, TracedTensor):
        raise TypeError(
            f"traced function must return a single tensor, got {type(result)!r}"
        )

    return_value = result.mlir_value
    _finalize_signature(
        scope.module, scope.func_op, scope.input_types, return_value, scope.ctx
    )
    scope.module.operation.verify()
    return scope.module, return_value.type
