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
    IntegerType,
    Attribute,
    DictAttr,
)

import ttnn

from ttnn_jit._src.tracing_compiler import JitContext
from ttnn_jit._src.conversions import (
    mlir_dtype_from_ttnn_dtype,
    ttnn_dtype_from_mlir_dtype,
)


def _traced_element_type(dtype, ctx):
    """MLIR element type for a traced tensor of the given ttnn dtype.

    Same as the shared mapping, except ttnn.int32 -> signed `si32`. The analysis
    pipeline (`ttnn-layout`) canonicalizes int32 tensor layouts to `si32`, and
    the optimizer's ScalarDataTypeAnalysis asserts the tensor's element type
    matches its layout's scalar type -- so an int32 tensor traced as signless
    `i32` (what the shared jit/full-pipeline mapping emits) would trip that
    assert once it survives to the optimizer (e.g. a paged_fill_cache page
    table kept alive by the keep-alive anchor). This override is advisor-only;
    the shared `mlir_dtype_from_ttnn_dtype` (used by @jit) is left untouched.
    """
    if dtype == ttnn.int32:
        return IntegerType.get_signed(32, ctx)
    return mlir_dtype_from_ttnn_dtype(dtype, ctx)


class TracedTensor:
    """Proxy standing in for a tensor during tracing.

    Holds the MLIR Value for the tensor and exposes shape/dtype (read from the
    MLIR result type) so the traced model's own Python shape math works. Layout
    is intentionally unknown -- the optimizer assigns it.
    """

    def __init__(self, mlir_value):
        self.mlir_value = mlir_value

    @property
    def __class__(self):
        # Spoof isinstance(proxy, ttnn.Tensor) so real model code that gates on
        # `isinstance(x, ttnn.Tensor)` accepts the proxy. `type(proxy)` still
        # returns the real TracedTensor class, so our internal checks use
        # `type(x) is TracedTensor` (which ignores this spoof).
        import ttnn

        return ttnn.Tensor

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
            RankedTensorType.get(list(shape), _traced_element_type(dtype, ctx))
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


def _restore_patched(module, saved):
    """Restore attrs on `module` from a {name: original_or__MISSING} mapping.

    `_MISSING` means the attr didn't exist before patching, so it's deleted
    (rather than set to a sentinel) to leave the module exactly as found.
    """
    for name, original in saved.items():
        if original is _MISSING:
            try:
                delattr(module, name)
            except AttributeError:
                pass
        else:
            setattr(module, name, original)


_PASSTHROUGH_IDENTITY = {
    "to_memory_config",
    "interleaved_to_sharded",
    "sharded_to_interleaved",
}
_PASSTHROUGH_NONE = {"deallocate"}


def _identity_passthrough(*args, **kwargs):
    # Layout op: return the first tensor operand unchanged; emit no TTIR.
    for a in args:
        if type(a) is TracedTensor:
            return a
    for v in kwargs.values():
        if type(v) is TracedTensor:
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
            shape, _traced_element_type(tensor.dtype, jit_ctx.ctx)
        )
        value = ttir.EmptyOp(ttype).result
    cache[key] = value
    return value


def _capture(arg, jit_ctx):
    if isinstance(arg, (list, tuple)):
        return type(arg)(_capture(a, jit_ctx) for a in arg)
    if type(arg) is TracedTensor or isinstance(arg, (int, float, bool)) or arg is None:
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


def _broadcast_batch(x, y):
    # numpy-style right-aligned broadcast of two batch-dim lists.
    n = max(len(x), len(y))
    xr = [1] * (n - len(x)) + list(x)
    yr = [1] * (n - len(y)) + list(y)
    return [(a if b == 1 else b if a == 1 else max(a, b)) for a, b in zip(xr, yr)]


def _linear_handler(jit_ctx, a, b, *, bias=None, dtype=None, **kwargs):
    a_type = a.mlir_value.type
    b_type = b.mlir_value.type
    a_shape = [int(d) for d in a_type.shape]
    b_shape = [int(d) for d in b_type.shape]
    # matmul/linear output = broadcast(a[:-2], b[:-2]) + [a[-2], b[-1]]
    # (matches ttir.LinearOp verifier; naive a[:-1]+[b[-1]] mis-ranks when a/b
    # have different batch-dim counts).
    out_shape = _broadcast_batch(a_shape[:-2], b_shape[:-2]) + [
        a_shape[-2],
        b_shape[-1],
    ]
    elem_type = (
        _traced_element_type(dtype, jit_ctx.ctx)
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
            _traced_element_type(dtype, jit_ctx.ctx),
        )
        return ttir.typecast(result=result_type, input=x.mlir_value)


def _rms_norm_handler(jit_ctx, x, *, epsilon=1e-5, weight=None, **kwargs):
    x_type = x.mlir_value.type
    hidden = int(x_type.shape[-1])
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get(
            [int(d) for d in x_type.shape], x_type.element_type
        )
        weight_val = None
        if weight is not None:
            w_type = weight.mlir_value.type
            w_shape = [int(d) for d in w_type.shape]
            # The TTIR rms_norm verifier requires weight.shape == normalized_shape
            # == [hidden]. TTNN models tile-pack the norm weight (e.g. [1,1,H/32,32]);
            # flatten it to [hidden] for the analysis graph (values are irrelevant).
            if w_shape != [hidden]:
                flat_type = RankedTensorType.get([hidden], w_type.element_type)
                weight_val = ttir.reshape(
                    result=flat_type, input=weight.mlir_value, shape=[hidden]
                )
            else:
                weight_val = weight.mlir_value
        eps_attr = FloatAttr.get(F32Type.get(jit_ctx.ctx), float(epsilon))
        return ttir.rms_norm(
            result=result_type,
            input=x.mlir_value,
            normalized_shape=[hidden],
            weight=weight_val,
            epsilon=eps_attr,
        )


def _reshape_handler(jit_ctx, x, shape=None, **kwargs):
    # ttnn.reshape(x, shape) — shape is positional (args[1]) or a `shape=` kwarg,
    # a list/tuple/ttnn.Shape that may contain a single -1 to infer. Overrides the
    # jit_functions reshape handler, which mis-resolves the shape as an operand.
    if shape is None:
        shape = kwargs.get("shape")
    dims = [int(d) for d in shape]
    in_shape = [int(d) for d in x.mlir_value.type.shape]
    if -1 in dims:
        total = 1
        for d in in_shape:
            total *= d
        known = 1
        for d in dims:
            if d != -1:
                known *= d
        dims[dims.index(-1)] = total // known
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get(dims, x.mlir_value.type.element_type)
        return ttir.reshape(result=result_type, input=x.mlir_value, shape=dims)


_VALUE_HANDLERS = {
    "linear": _linear_handler,
    "typecast": _typecast_handler,
    "rms_norm": _rms_norm_handler,
    "reshape": _reshape_handler,
}


def _nlp_create_qkv_heads_handler(
    jit_ctx, xqkv, *, num_heads, num_kv_heads, transpose_k_heads=False, **kwargs
):
    """Split a fused QKV projection into per-head Q/K/V tensors.

    The TTIR SplitQueryKeyValueAndSplitHeadsOp verifier requires a rank-3
    input [batch, seq, qkv_size], but the model passes a rank-4 fused qkv
    tensor [batch, 1, seq, qkv_size] (the singleton dim from an un-squeezed
    linear projection). Reshape it down to rank-3 first.
    """
    t = xqkv.mlir_value.type
    b = int(t.shape[0])
    seq = int(t.shape[-2])
    qkv_size = int(t.shape[-1])
    head_dim = qkv_size // (num_heads + 2 * num_kv_heads)
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        et = t.element_type
        reshaped_type = RankedTensorType.get([b, seq, qkv_size], et)
        reshaped = ttir.reshape(reshaped_type, xqkv.mlir_value, [b, seq, qkv_size])
        qt = RankedTensorType.get([b, num_heads, seq, head_dim], et)
        kt = RankedTensorType.get([b, num_kv_heads, seq, head_dim], et)
        vt = RankedTensorType.get([b, num_kv_heads, seq, head_dim], et)
        return ttir.split_query_key_value_and_split_heads(
            qt,
            kt,
            vt,
            reshaped,
            num_heads,
            bool(transpose_k_heads),
            num_kv_heads=num_kv_heads,
        )


_EXPERIMENTAL_MULTI = {
    "nlp_create_qkv_heads": _nlp_create_qkv_heads_handler,
}


def _nlp_concat_heads_handler(jit_ctx, x, **kwargs):
    """Merge per-head attention output back into a single hidden dimension.

    The real ttnn.experimental.nlp_concat_heads shuffles
    [b, num_heads, seq, head_dim] into [b, 1, seq, num_heads*head_dim] (a
    singleton "user" dim retained for its internal tiling convention). The
    TTIR ConcatenateHeadsOp verifier instead requires a rank-3 output
    [b, seq, num_heads*head_dim] -- no singleton dim. Emit that TTIR shape;
    the singleton-dim discrepancy is a runtime-only quirk of the ttnn op
    that the TTIR/TTNN dialect abstraction doesn't model.
    """
    t = x.mlir_value.type
    b = int(t.shape[0])
    nh = int(t.shape[1])
    seq = int(t.shape[2])
    hd = int(t.shape[3])
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get([b, seq, nh * hd], t.element_type)
        return ttir.concatenate_heads(result=result_type, input=x.mlir_value)


def _paged_fill_cache_handler(jit_ctx, cache, input, page_table, **kwargs):
    """Fill a paged KV cache in place; result models the updated cache.

    Real call sites pass a scalar `batch_idx` (int) kwarg, not a tensor; the
    TTIR PagedFillCacheOp only models an optional `batch_idx_tensor` operand
    (per-row tensor), so `batch_idx` and other real-op-only kwargs (e.g.
    `block_size`, `cache_position_modulo`) are silently ignored here -- they
    don't affect the shapes/dtypes the layout optimizer analyzes.
    """
    t = cache.mlir_value.type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get([int(d) for d in t.shape], t.element_type)
        return ttir.paged_fill_cache(
            result=result_type,
            cache=cache.mlir_value,
            input=input.mlir_value,
            page_table=page_table.mlir_value,
        )


def _rotary_embedding_llama_handler(
    jit_ctx, input, cos_cache, sin_cache, trans_mat, *, is_decode_mode=False, **kwargs
):
    """Apply RoPE to a Q/K head tensor; result shape matches `input`."""
    t = input.mlir_value.type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get([int(d) for d in t.shape], t.element_type)
        return ttir.rotary_embedding_llama(
            result=result_type,
            input=input.mlir_value,
            cos_cache=cos_cache.mlir_value,
            sin_cache=sin_cache.mlir_value,
            trans_mat=trans_mat.mlir_value,
            is_decode_mode=bool(is_decode_mode),
        )


_EXPERIMENTAL_VALUE = {
    "nlp_concat_heads": _nlp_concat_heads_handler,
    "paged_fill_cache": _paged_fill_cache_handler,
    "rotary_embedding_llama": _rotary_embedding_llama_handler,
}


def _sdpa_handler(jit_ctx, q, k, v, *, is_causal=None, scale=None, **kwargs):
    t = q.mlir_value.type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get([int(d) for d in t.shape], t.element_type)
        return ttir.scaled_dot_product_attention(
            result=result_type,
            query=q.mlir_value,
            key=k.mlir_value,
            value=v.mlir_value,
            is_causal=is_causal,
            scale=scale,
        )


def _chunked_sdpa_handler(
    jit_ctx,
    *,
    input_tensor_q,
    input_tensor_k,
    input_tensor_v,
    page_table_tensor=None,
    scale=None,
    **kwargs,
):
    t = input_tensor_q.mlir_value.type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        result_type = RankedTensorType.get([int(d) for d in t.shape], t.element_type)
        # chunked SDPA maps to the same TTIR SDPA op for layout analysis; the
        # chunk/page mechanics don't change the output layout the optimizer sees.
        return ttir.scaled_dot_product_attention(
            result=result_type,
            query=input_tensor_q.mlir_value,
            key=input_tensor_k.mlir_value,
            value=input_tensor_v.mlir_value,
            is_causal=True,
            scale=scale,
        )


_TRANSFORMER_VALUE = {
    "scaled_dot_product_attention": _sdpa_handler,
    "chunked_scaled_dot_product_attention": _chunked_sdpa_handler,
}


@contextmanager
def patch_ttnn(jit_ctx):
    """Monkeypatch allowlisted ttnn.<op> to build TTIR; restore on exit."""
    import ttnn

    namespace = TTNNJitNamespaceUpdater(jit_ctx)
    originals = {}
    experimental = getattr(ttnn, "experimental", None)
    exp_originals = {}
    transformer = getattr(ttnn, "transformer", None)
    transformer_originals = {}
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
            setattr(
                ttnn, "mul", _make_traced_op(getattr(namespace, "multiply"), jit_ctx)
            )
        if experimental is not None:
            for name, fn in _EXPERIMENTAL_MULTI.items():
                exp_originals[name] = getattr(experimental, name, _MISSING)
                setattr(experimental, name, _make_traced_multi_op(fn, jit_ctx))
            for name, value_fn in _EXPERIMENTAL_VALUE.items():
                exp_originals[name] = getattr(experimental, name, _MISSING)
                setattr(experimental, name, _make_traced_value_op(value_fn, jit_ctx))
        if transformer is not None:
            for name, value_fn in _TRANSFORMER_VALUE.items():
                transformer_originals[name] = getattr(transformer, name, _MISSING)
                setattr(transformer, name, _make_traced_value_op(value_fn, jit_ctx))
        yield
    finally:
        _restore_patched(ttnn, originals)
        if experimental is not None:
            _restore_patched(experimental, exp_originals)
        if transformer is not None:
            _restore_patched(transformer, transformer_originals)


def _result_is_unused(value):
    return next(iter(value.uses), None) is None


def _collect_keepalive_anchors(block, return_value):
    """Traced-op results that nothing consumes.

    A traced op whose result is never used is `Pure` at the TTIR level (all
    TTIR_NamedOps are), so `ttnn-layout`'s greedy DCE deletes it before the
    optimizer ever sees it -- e.g. an in-place `ttir.paged_fill_cache` whose
    returned cache handle isn't threaded anywhere. Returning these as extra
    function outputs makes them "used" so the whole traced graph survives and
    the advisor reports layouts for every op the user wrote. `ttir.empty`
    scaffolding is skipped (pure placeholders, nothing to advise on).
    """
    anchors = []
    for op in block.operations:
        if op.name == "ttir.empty":
            continue
        for result in op.results:
            if result != return_value and _result_is_unused(result):
                anchors.append(result)
    return anchors


def _lift_weights_to_args(old_block, weight_values, ctx):
    """Replace captured-weight ttir.empty placeholders with function arguments.

    The tracer materializes captured tensors (weights, KV caches, page tables,
    rope tables) as ttir.empty. Lifting them to real function arguments lets the
    optimizer model them as DRAM-resident parameters (as in a real compile)
    instead of freshly-allocated intermediates. Returns the appended arg types
    in order so the caller can extend the function signature.
    """
    loc = Location.unknown(ctx)
    weight_types = []
    empties = []
    for wv in weight_values:
        weight_types.append(wv.type)
        new_arg = old_block.add_argument(wv.type, loc)
        empties.append(wv.owner)
        wv.replace_all_uses_with(new_arg)
    for empty_op in empties:
        empty_op.erase()
    return weight_types


def _finalize_signature(module, func_op, input_types, return_value, ctx, weight_values):
    """Add the return op and rebuild the func with the correct signature.

    Mirrors TracingCompiler._update_function_signature: MLIR can't mutate a func
    signature in place, so recreate the func and move the body block. Captured
    weights are lifted to trailing function arguments (tagged as parameters);
    the original traced activations stay as the leading arguments (tagged input).
    """
    old_block = func_op.regions[0].blocks[0]
    weight_types = _lift_weights_to_args(old_block, weight_values, ctx)

    results = [return_value, *_collect_keepalive_anchors(old_block, return_value)]
    if not old_block.operations or not isinstance(
        old_block.operations[-1], func.ReturnOp
    ):
        with InsertionPoint(old_block), Location.unknown(ctx):
            func.ReturnOp(results)

    all_input_types = list(input_types) + weight_types
    with InsertionPoint(module.body), Location.unknown(ctx):
        new_func = func.FuncOp(
            name=func_op.name.value,
            type=(all_input_types, [r.type for r in results]),
        )
    old_block.append_to(new_func.regions[0])
    func_op.erase()

    # Tag args: leading traced activations as <input>, lifted weights as
    # <parameter>, so the optimizer can distinguish weights from activations.
    n_inputs = len(input_types)
    input_attr = Attribute.parse("#ttcore.argument_type<input>", ctx)
    param_attr = Attribute.parse("#ttcore.argument_type<parameter>", ctx)
    new_func.arg_attrs = [
        DictAttr.get(
            {"ttcore.argument_type": input_attr if i < n_inputs else param_attr}, ctx
        )
        for i in range(len(all_input_types))
    ]


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
