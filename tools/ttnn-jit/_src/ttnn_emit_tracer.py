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

Coverage is the full transformer-decoder vocabulary: dense compute
(matmul/linear, elementwise binary/unary, rms_norm, softmax), data movement
(slice, reshape, transpose, permute, concat, embedding), attention (SDPA,
paged SDPA decode), heads (QKV split + concat, prefill and decode), RoPE, and
in-place paged KV-cache ops, the MoE router (topk/scatter/zeros/arange/pad/
clamp/fill_cache) and the routed-expert ttnn.sparse_matmul. The full Llama
decoder sweeps in both phases, and a sparse-MoE decoder traces end to end.

Note on sparse_matmul: TTNN_SparseMatmulOp is OpModelExempt, so the optimizer
returns notImplemented for it and places nothing on the op itself. That is a
soft gap, not a terminal -- tracing it still lets the optimizer see and place
the surrounding graph (router, activations, the MoE tail), which is the point.
Not yet ported: pow/pow_tensor/rearrange, allowlisted ops the TTIR tracer
covers via BaseOpHandler and this tracer stubs via _unhandled.
"""

from contextlib import contextmanager

from ttnn_jit.ttmlir.dialects import func, ttnn, ttcore
from ttnn_jit.ttmlir.ir import (
    F32Type,
    FloatAttr,
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
from ttnn_jit._src.supported_ops import (
    unary_ops,
    binary_ops,
    reduction_ops,
    tm_ops,
    data_movement_ops,
)

# The @jit allowlist. The TTIR tracer routes all of these through BaseOpHandler;
# the direct-TTNN tracer has an explicit handler per op, so any allowlist op
# without one is stubbed to fail loudly (see _unhandled) rather than silently
# fall through to a real on-device ttnn call.
_ALLOWLIST = (
    set(unary_ops)
    | set(binary_ops)
    | set(reduction_ops)
    | set(tm_ops)
    | set(data_movement_ops)
    | {"matmul", "div", "pow"}
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
    # Unsigned integers reach the tracer via index/position tensors (gpt-oss
    # feeds a ui32 cache position and ui16 router indices).
    if s in ("ui32", "u32"):
        return ttcore.DataType.UInt32
    if s in ("ui16", "u16"):
        return ttcore.DataType.UInt16
    if s in ("ui8", "u8"):
        return ttcore.DataType.UInt8
    if s == "i1":
        return ttcore.DataType.Bool
    if s == "f16":
        return ttcore.DataType.Float16
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
    # In-place ops that carry no dialect op (ttnn.copy) rebind their destination
    # here. Checked BEFORE weight_cache so the placeholder stays in weight_cache
    # for _finalize_signature to lift and erase, while reads see the update.
    # Clobbering weight_cache instead orphans the placeholder, and a surviving
    # ttir.empty aborts the pipeline with "Backend constraints are not
    # implemented for op ttir.empty".
    alias = getattr(jit_ctx, "cache_alias", None)
    if alias is not None and key in alias:
        return alias[key]
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


def _sparse_matmul_handler(
    jit_ctx,
    a,
    b,
    *,
    sparsity=None,
    is_input_a_sparse=None,
    is_input_b_sparse=None,
    nnz=None,
    **kwargs,
):
    # ttnn.sparse_matmul(a, b=[1,E,K,N], sparsity) -> ttnn.sparse_matmul. Output
    # shape per SparseMatmulOp::verify (E,K,N from b; M = a[-2]): dense-sparse
    # (b sparse, e.g. MoE gate/up) -> [A,B,1,E,M,N]; sparse-dense (a sparse, e.g.
    # down) -> [A,B,M,N]; both sparse -> [1,E,M,N]. ttnn defaults to b-sparse when
    # neither flag is set. program_config/compute_config are dropped: every
    # direct-TTNN tensor starts DRAM-interleaved and the optimizer re-decides.
    a_sparse = bool(is_input_a_sparse)
    b_sparse = (
        bool(is_input_b_sparse) if is_input_b_sparse is not None else not a_sparse
    )
    a_shape = [int(d) for d in a.mlir_value.type.shape]
    b_shape = [int(d) for d in b.mlir_value.type.shape]
    E, N, M = b_shape[1], b_shape[-1], a_shape[-2]
    if a_sparse and b_sparse:
        out = [1, E, M, N]
    elif b_sparse:
        out = [a_shape[0], a_shape[1], 1, E, M, N]
    else:
        out = [a_shape[0], a_shape[1], M, N]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, a.mlir_value, out)
        return ttnn.sparse_matmul(
            result=rt,
            a=a.mlir_value,
            b=b.mlir_value,
            sparsity=sparsity.mlir_value,
            is_input_a_sparse=a_sparse,
            is_input_b_sparse=b_sparse,
            nnz=nnz,
        )


def _softplus_handler(jit_ctx, x, **kwargs):
    """``ttnn.softplus(x)`` -> ``log(exp(x) + 1)``.

    DECOMPOSITION, not 1:1. The TTNN dialect has no standalone softplus op --
    `SoftPlus` exists only as a `UnaryOpType` enum case usable as a matmul fused
    activation, so there is nothing to emit directly. Three ops stand in for one,
    which is faithful in value but changes what the optimizer counts. Replace this
    with a named TTNN_SoftplusOp if op-for-op fidelity starts to matter.
    """
    dims = [int(d) for d in x.mlir_value.type.shape]
    elem = x.mlir_value.type.element_type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _tt(jit_ctx.ctx, dims, elem)
        e = ttnn.exp(result=rt, input=x.mlir_value)
        one = ttnn.ones(
            result=_tt(jit_ctx.ctx, dims, elem),
            shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, dims),
        )
        s = ttnn.add(result=_tt(jit_ctx.ctx, dims, elem), lhs=e, rhs=one)
        return ttnn.log(result=_tt(jit_ctx.ctx, dims, elem), input=s)


def _repeat_interleave_handler(jit_ctx, x, repeats=None, dim=None, **kwargs):
    """``ttnn.repeat_interleave(x, repeats, dim)`` -> ``ttnn.repeat_interleave``."""
    repeats = int(kwargs.get("repeats", repeats))
    dim = int(kwargs.get("dim", dim))
    dims = [int(d) for d in x.mlir_value.type.shape]
    if dim < 0:
        dim += len(dims)
    out = list(dims)
    out[dim] = dims[dim] * repeats
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        return ttnn.repeat_interleave(
            result=_retype(jit_ctx.ctx, x.mlir_value, out),
            input=x.mlir_value,
            repeats=repeats,
            dim=dim,
        )


def _pow_handler(jit_ctx, x, y, **kwargs):
    """``ttnn.pow(x, y)`` -> ``ttnn.pow_tensor`` (tensor y) or ``ttnn.pow_scalar``."""
    dims = [int(d) for d in x.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, dims)
        if hasattr(y, "mlir_value"):
            return ttnn.pow_tensor(result=rt, lhs=x.mlir_value, rhs=y.mlir_value)
        return ttnn.pow_scalar(
            result=rt,
            lhs=x.mlir_value,
            rhs=FloatAttr.get(F32Type.get(jit_ctx.ctx), float(y)),
        )


def _ones_like_handler(jit_ctx, input, **kwargs):
    """``ttnn.ones_like(x)`` -> ``ttnn.ones`` of x's shape/element type."""
    dims = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        return ttnn.ones(
            result=_retype(jit_ctx.ctx, input.mlir_value, dims),
            shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, dims),
        )


def _binary(op_fn):
    def handler(jit_ctx, x, y, **kwargs):
        # Either operand may be a python scalar -- `ttnn.multiply(gate, 1.703125)`
        # is ordinary model code (gpt-oss's SwiGLU alpha). Materialize it as a
        # ttnn.full shaped like the tensor operand so the graph stays all-tensor,
        # the same way _where_handler does.
        tensor_ref = x if hasattr(x, "mlir_value") else y
        if not hasattr(tensor_ref, "mlir_value"):
            raise TypeError("binary op needs at least one tensor operand")
        with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
            def dims_of(v):
                if hasattr(v, "mlir_value"):
                    return [int(d) for d in v.mlir_value.type.shape]
                return [int(d) for d in tensor_ref.mlir_value.type.shape]

            xs, ys = dims_of(x), dims_of(y)
            n = max(len(xs), len(ys))
            xr = [1] * (n - len(xs)) + xs
            yr = [1] * (n - len(ys)) + ys
            out = [a if b == 1 else b if a == 1 else max(a, b) for a, b in zip(xr, yr)]
            elem = tensor_ref.mlir_value.type.element_type

            def operand(v):
                if hasattr(v, "mlir_value"):
                    return v.mlir_value
                return ttnn.full(
                    result=_tt(jit_ctx.ctx, out, elem),
                    shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, out),
                    fill_value=FloatAttr.get(F32Type.get(jit_ctx.ctx), float(v)),
                )

            rt = _tt(jit_ctx.ctx, out, elem)
            return op_fn(result=rt, lhs=operand(x), rhs=operand(y))

    return handler


def _unary(op_fn):
    def handler(jit_ctx, x, **kwargs):
        with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
            shape = [int(d) for d in x.mlir_value.type.shape]
            rt = _retype(jit_ctx.ctx, x.mlir_value, shape)
            return op_fn(result=rt, input=x.mlir_value)

    return handler


def _reshape_handler(jit_ctx, x, shape=None, padded_shape=None, **kwargs):
    # shape may carry a single -1 to infer; the decode path also passes a second
    # (tile-padded) shape -- model the logical shape, ignore padding.
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
    hidden = int(x.mlir_value.type.shape[-1])
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        shape = [int(d) for d in x.mlir_value.type.shape]
        rt = _retype(jit_ctx.ctx, x.mlir_value, shape)
        weight_val = None
        if weight is not None:
            # ttnn.rms_norm requires a 1D [hidden] weight; TTNN models tile-pack
            # the norm weight (e.g. [1,1,H/32,32]), so flatten it for the graph.
            w_type = weight.mlir_value.type
            if [int(d) for d in w_type.shape] != [hidden]:
                weight_val = ttnn.reshape(
                    result=_tt(jit_ctx.ctx, [hidden], w_type.element_type),
                    input=weight.mlir_value,
                    shape=[hidden],
                )
            else:
                weight_val = weight.mlir_value
        return ttnn.rms_norm(
            result=rt, input=x.mlir_value, weight=weight_val, epsilon=float(epsilon)
        )


def _where_handler(jit_ctx, predicate, true_value, false_value, **kwargs):
    """Emit ternary selection with normal right-aligned broadcasting.

    Either branch may be a python scalar (`ttnn.where(mask, 0.0, -3.4e38)` is
    the standard decode attention-mask idiom); a scalar branch is materialized
    as a ttnn.full of the broadcast shape so the graph stays all-tensor.
    """

    branches = (predicate, true_value, false_value)
    shapes = [
        [int(d) for d in v.mlir_value.type.shape]
        for v in branches
        if hasattr(v, "mlir_value")
    ]
    out = shapes[0]
    for shape in shapes[1:]:
        out = _broadcast_batch(out, shape)

    # Element type comes from a tensor branch when there is one, else from the
    # predicate (which the producing comparison already emitted in the model's
    # activation dtype).
    elem_source = next(
        (v for v in (true_value, false_value) if hasattr(v, "mlir_value")), predicate
    )
    elem = elem_source.mlir_value.type.element_type

    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):

        def operand(value):
            if hasattr(value, "mlir_value"):
                return value.mlir_value
            return ttnn.full(
                result=_tt(jit_ctx.ctx, out, elem),
                shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, out),
                fill_value=FloatAttr.get(F32Type.get(jit_ctx.ctx), float(value)),
            )

        rt = _tt(jit_ctx.ctx, out, elem)
        return ttnn.where(
            result=rt,
            first=predicate.mlir_value,
            second=operand(true_value),
            third=operand(false_value),
        )


def _slice_handler(jit_ctx, x, starts=None, ends=None, steps=None, **kwargs):
    if starts is None:
        starts = kwargs.get("slice_start", kwargs.get("starts"))
    if ends is None:
        ends = kwargs.get("slice_end", kwargs.get("ends"))
    if steps is None:
        steps = kwargs.get("slice_step", kwargs.get("steps"))
    in_shape = [int(d) for d in x.mlir_value.type.shape]
    # Resolve Python-style negative / open-ended indices against the input dims
    # (the layout builder rejects negative shapes, unlike the lazy TTIR path).
    begins, ends_i, step = [], [], []
    for i in range(len(starts)):
        dim = in_shape[i]
        s = int(starts[i])
        e = int(ends[i])
        st = 1 if steps is None else int(steps[i])
        if s < 0:
            s += dim
        if e < 0:
            e += dim
        e = min(e, dim)
        begins.append(s)
        ends_i.append(e)
        step.append(st)
    out = [(ends_i[i] - begins[i] + step[i] - 1) // step[i] for i in range(len(begins))]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, out)
        return ttnn.slice_static(
            result=rt, input=x.mlir_value, begins=begins, ends=ends_i, step=step
        )


def _split_handler(jit_ctx, x, split_size, dim=0, **kwargs):
    """Emit a runtime ``split`` boundary as static TTNN slices."""

    shape = [int(d) for d in x.mlir_value.type.shape]
    axis = int(dim) % len(shape)
    axis_size = shape[axis]
    if isinstance(split_size, (list, tuple)):
        sizes = [int(size) for size in split_size]
        if sum(sizes) != axis_size:
            raise ValueError(
                f"split sizes {sizes} do not cover dimension {axis_size}"
            )
    else:
        chunk = int(split_size)
        if chunk <= 0:
            raise ValueError(f"split_size must be positive, got {chunk}")
        sizes = [min(chunk, axis_size - start) for start in range(0, axis_size, chunk)]

    results = []
    start = 0
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        for size in sizes:
            begins = [0] * len(shape)
            ends = list(shape)
            begins[axis] = start
            ends[axis] = start + size
            out = list(shape)
            out[axis] = size
            results.append(
                ttnn.slice_static(
                    result=_retype(jit_ctx.ctx, x.mlir_value, out),
                    input=x.mlir_value,
                    begins=begins,
                    ends=ends,
                    step=[1] * len(shape),
                )
            )
            start += size
    return results


def _unsqueeze_to_4d_handler(jit_ctx, x, **kwargs):
    shape = [int(d) for d in x.mlir_value.type.shape]
    dims = [1] * (4 - len(shape)) + shape if len(shape) < 4 else shape
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, dims)
        return ttnn.reshape(result=rt, input=x.mlir_value, shape=dims)


def _transpose_handler(jit_ctx, x, dim0, dim1, **kwargs):
    shape = [int(d) for d in x.mlir_value.type.shape]
    d0, d1 = int(dim0) % len(shape), int(dim1) % len(shape)
    out = list(shape)
    out[d0], out[d1] = out[d1], out[d0]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, out)
        return ttnn.transpose(result=rt, input=x.mlir_value, dim0=d0, dim1=d1)


def _permute_handler(jit_ctx, x, permutation=None, dims=None, **kwargs):
    perm = [int(p) for p in (permutation if permutation is not None else dims)]
    shape = [int(d) for d in x.mlir_value.type.shape]
    out = [shape[p] for p in perm]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, out)
        return ttnn.permute(result=rt, input=x.mlir_value, permutation=perm)


def _concat_handler(jit_ctx, tensors, dim=0, **kwargs):
    vals = [t.mlir_value for t in tensors]
    shapes = [[int(d) for d in v.type.shape] for v in vals]
    d = int(dim) % len(shapes[0])
    out = list(shapes[0])
    out[d] = sum(s[d] for s in shapes)
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _tt(jit_ctx.ctx, out, vals[0].type.element_type)
        return ttnn.concat(result=rt, inputs=vals, dim=d)


def _embedding_handler(jit_ctx, indices, weight, **kwargs):
    ishape = [int(d) for d in indices.mlir_value.type.shape]
    hidden = int(weight.mlir_value.type.shape[-1])
    out = ishape + [hidden]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _tt(jit_ctx.ctx, out, weight.mlir_value.type.element_type)
        return ttnn.embedding(
            result=rt, input=indices.mlir_value, weight=weight.mlir_value
        )


def _rotary_embedding_llama_handler(
    jit_ctx, input, cos_cache, sin_cache, trans_mat, *, is_decode_mode=False, **kwargs
):
    shape = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, input.mlir_value, shape)
        return ttnn.rotary_embedding_llama(
            result=rt,
            input=input.mlir_value,
            cos_cache=cos_cache.mlir_value,
            sin_cache=sin_cache.mlir_value,
            trans_mat=trans_mat.mlir_value,
            is_decode_mode=bool(is_decode_mode),
        )


def _rotary_embedding_handler(
    jit_ctx, input, cos_cache, sin_cache, token_index=None, **kwargs
):
    """Apply token-indexed RoPE; result shape matches ``input``."""
    shape = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, input.mlir_value, shape)
        return ttnn.rotary_embedding(
            result=rt,
            input=input.mlir_value,
            cos_cache=cos_cache.mlir_value,
            sin_cache=sin_cache.mlir_value,
            token_index=(None if token_index is None else int(token_index)),
        )


def _rotary_embedding_hf_handler(
    jit_ctx, input, cos_cache, sin_cache, *, is_decode_mode=None, **kwargs
):
    """``ttnn.experimental.rotary_embedding_hf`` -> ``ttnn.rotary_embedding``.

    Not a stand-in: the dialect op documents exactly the HuggingFace formula
    (``x*cos + rotate_half(x)*sin``, rotate_half swapping the halves of the last
    dim), which is what the HF runtime op computes. The runtime op additionally
    takes ``is_decode_mode`` to select the per-batch-position kernel variant --
    that changes neither shapes nor layouts, and the dialect op carries no such
    flag, so it is dropped. The traced graph stays shape- and layout-faithful,
    which is what the advisor consumes.
    """
    del is_decode_mode
    shape = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, input.mlir_value, shape)
        return ttnn.rotary_embedding(
            result=rt,
            input=input.mlir_value,
            cos_cache=cos_cache.mlir_value,
            sin_cache=sin_cache.mlir_value,
            token_index=None,
        )


# --- MoE / router + long-tail creation ops ---------------------------------
# gpt-oss's decoder is the first traced model needing these; every one maps 1:1
# onto an existing TTNN dialect op, same as the rest of this file.


def _zeros_handler(jit_ctx, shape=None, *, dtype=None, device=None, **kwargs):
    """``ttnn.zeros(shape)`` -> ``ttnn.zeros``. dtype defaults to bf16."""
    dims = [int(d) for d in (shape if shape is not None else kwargs.get("size", []))]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        elem = _traced_element_type(dtype or _ttnn_rt.bfloat16, jit_ctx.ctx)
        return ttnn.zeros(
            result=_tt(jit_ctx.ctx, dims, elem),
            shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, dims),
        )


def _zeros_like_handler(jit_ctx, input, **kwargs):
    """``ttnn.zeros_like(x)`` -> ``ttnn.zeros`` of x's shape/element type."""
    dims = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        return ttnn.zeros(
            result=_retype(jit_ctx.ctx, input.mlir_value, dims),
            shape=ttnn.ir.ShapeAttr.get(jit_ctx.ctx, dims),
        )


def _arange_handler(jit_ctx, start=0, end=None, step=1, *, dtype=None, **kwargs):
    """``ttnn.arange(start, end, step)`` -> ``ttnn.arange`` (1-D)."""
    if end is None:
        start, end = 0, start
    start, end, step = int(start), int(end), int(step or 1)
    n = max(0, -(-(end - start) // step))
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        elem = _traced_element_type(dtype or _ttnn_rt.bfloat16, jit_ctx.ctx)
        return ttnn.arange(
            result=_tt(jit_ctx.ctx, [n], elem), start=start, end=end, step=step
        )


def _clamp_handler(jit_ctx, input, min=None, max=None, **kwargs):
    """``ttnn.clamp`` -> clamp_scalar (bounds are scalars in every traced use)."""
    lo = kwargs.get("min", min)
    hi = kwargs.get("max", max)
    dims = [int(d) for d in input.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        f32 = F32Type.get(jit_ctx.ctx)
        return ttnn.clamp_scalar(
            result=_retype(jit_ctx.ctx, input.mlir_value, dims),
            input=input.mlir_value,
            min=FloatAttr.get(f32, float("-inf") if lo is None else float(lo)),
            max=FloatAttr.get(f32, float("inf") if hi is None else float(hi)),
        )


def _pad_handler(jit_ctx, input, padding=None, value=0.0, **kwargs):
    """``ttnn.pad`` -> ``ttnn.pad``; padding is (before, after) per dim."""
    padding = kwargs.get("padding", padding) or []
    dims = [int(d) for d in input.mlir_value.type.shape]
    flat, out = [], list(dims)
    for i, pair in enumerate(padding):
        before, after = (int(pair[0]), int(pair[1])) if isinstance(pair, (list, tuple)) else (0, int(pair))
        flat += [before, after]
        if i < len(out):
            out[i] += before + after
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        return ttnn.pad(
            result=_retype(jit_ctx.ctx, input.mlir_value, out),
            input=input.mlir_value,
            padding=flat,
            value=FloatAttr.get(F32Type.get(jit_ctx.ctx), float(value or 0.0)),
            use_multicore=True,
        )


def _scatter_handler(jit_ctx, input, dim=None, index=None, src=None, **kwargs):
    """``ttnn.scatter(input, dim=, index=, src=)`` -> ``ttnn.scatter``.

    The router scatters distinct expert indices into a zeros tensor, so a SUM
    reduce is equivalent to assignment (same modelling as the TTIR tracer).
    """
    dim = kwargs.get("dim", dim)
    index = kwargs.get("index", index)
    src = kwargs.get("src", src)
    dims = [int(d) for d in input.mlir_value.type.shape]
    d = 0 if dim is None else (int(dim) % len(dims))
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        return ttnn.scatter(
            result=_retype(jit_ctx.ctx, input.mlir_value, dims),
            input=input.mlir_value,
            index=index.mlir_value,
            source=src.mlir_value,
            dim=d,
            scatter_reduce_type=ttcore.ir.ReduceTypeAttr.get(
                jit_ctx.ctx, ttcore.ir.ReduceType.Sum
            ),
        )


def _topk_handler(jit_ctx, input, k=None, dim=None, largest=None, sorted=None, **kwargs):
    """``ttnn.topk`` -> ``ttnn.topk`` (values, indices); the top-k dim becomes k."""
    k = int(kwargs.get("k", k))
    dim = kwargs.get("dim", dim)
    dims = [int(d) for d in input.mlir_value.type.shape]
    d = (len(dims) - 1) if dim is None else (int(dim) % len(dims))
    out = list(dims)
    out[d] = k
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        vt = _retype(jit_ctx.ctx, input.mlir_value, out)
        # Indices are int32; use the shared mapping so the layout's scalar type
        # matches the tensor element type (ScalarDataTypeAnalysis asserts it).
        it = _tt(jit_ctx.ctx, out, _traced_element_type(_ttnn_rt.int32, jit_ctx.ctx))
        res = ttnn.topk(
            values=vt, indices=it, input_tensor=input.mlir_value, k=k, dim=d,
            largest=(True if largest is None else bool(largest)),
            sorted=(True if sorted is None else bool(sorted)),
        )
        return list(res)


def _fill_cache_handler(jit_ctx, cache, input, batch_offset=0, **kwargs):
    """``ttnn.fill_cache`` -> ``ttnn.fill_cache``; mutates the cache in place."""
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        ttnn.fill_cache(
            cache=cache.mlir_value,
            input=input.mlir_value,
            batch_offset=int(batch_offset or 0),
        )
    return cache.mlir_value


def _sdpa_handler(jit_ctx, q, k, v, *, is_causal=None, scale=None, **kwargs):
    shape = [int(d) for d in q.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, q.mlir_value, shape)
        return ttnn.scaled_dot_product_attention(
            result=rt,
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
    # chunked SDPA is layout-equivalent to plain SDPA for the optimizer; the
    # chunk/page mechanics don't change the output layout (mirrors TTIR tracer).
    shape = [int(d) for d in input_tensor_q.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, input_tensor_q.mlir_value, shape)
        return ttnn.scaled_dot_product_attention(
            result=rt,
            query=input_tensor_q.mlir_value,
            key=input_tensor_k.mlir_value,
            value=input_tensor_v.mlir_value,
            is_causal=True,
            scale=scale,
        )


def _paged_sdpa_decode_handler(
    jit_ctx, q, k, v, *, page_table_tensor, cur_pos_tensor=None, scale=None, **kwargs
):
    # Decode-phase paged attention; output shape matches the query [1,B,Hq,D].
    # ttnn's op is non-DPS (no `output` operand, unlike the TTIR op).
    shape = [int(d) for d in q.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, q.mlir_value, shape)
        return ttnn.paged_scaled_dot_product_attention_decode(
            result=rt,
            query=q.mlir_value,
            key=k.mlir_value,
            value=v.mlir_value,
            page_table=page_table_tensor.mlir_value,
            cur_pos_tensor=(
                cur_pos_tensor.mlir_value if cur_pos_tensor is not None else None
            ),
            scale=scale,
        )


def _sdpa_decode_handler(
    jit_ctx,
    q,
    k,
    v,
    *,
    is_causal=True,
    attn_mask=None,
    cur_pos_tensor=None,
    attention_sink=None,
    scale=None,
    **kwargs,
):
    """Dense-cache decode attention; output shape matches the query."""
    shape = [int(d) for d in q.mlir_value.type.shape]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, q.mlir_value, shape)
        return ttnn.scaled_dot_product_attention_decode(
            result=rt,
            query=q.mlir_value,
            key=k.mlir_value,
            value=v.mlir_value,
            is_causal=bool(is_causal),
            attention_mask=(attn_mask.mlir_value if attn_mask is not None else None),
            cur_pos_tensor=(cur_pos_tensor.mlir_value if cur_pos_tensor is not None else None),
            attention_sink=(attention_sink.mlir_value if attention_sink is not None else None),
            scale=scale,
        )


def _nlp_concat_heads_decode_handler(jit_ctx, x, *, num_heads=None, **kwargs):
    # Decode head merge: [1, B, Hq, D] -> [1, 1, B, Hq*D] (op requires 4D output).
    shp = [int(d) for d in x.mlir_value.type.shape]
    batch, heads, head_dim = shp[-3], shp[-2], shp[-1]
    nh = int(num_heads) if num_heads is not None else heads
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _tt(
            jit_ctx.ctx, [1, 1, batch, heads * head_dim], x.mlir_value.type.element_type
        )
        return ttnn.nlp_concat_heads_decode(result=rt, input=x.mlir_value, num_heads=nh)


def _nlp_concat_heads_handler(jit_ctx, x, **kwargs):
    # Prefill head merge: [b, nh, seq, hd] -> [b, seq, nh*hd]. Use concatenate_heads
    # (rank-3 output, no singleton dim) -- matches the op-model, like the TTIR path.
    # ttnn.nlp_concat_heads produces [b,1,seq,nh*hd], which its op-model rejects.
    t = x.mlir_value.type
    b, nh, seq, hd = (int(t.shape[i]) for i in range(4))
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _tt(jit_ctx.ctx, [b, seq, nh * hd], t.element_type)
        return ttnn.concatenate_heads(result=rt, input=x.mlir_value)


# --- multi-result handlers (return an OpResultList) -------------------------


def _nlp_create_qkv_heads_handler(
    jit_ctx, xqkv, *, num_heads, num_kv_heads, transpose_k_heads=False, **kwargs
):
    # Prefill QKV split. The op needs a rank-3 [b, seq, qkv] input; the model
    # passes rank-4 [b, 1, seq, qkv], so reshape down first (a real op, not a
    # decomposition of the split itself).
    t = xqkv.mlir_value.type
    b = int(t.shape[0])
    seq = int(t.shape[-2])
    qkv_size = int(t.shape[-1])
    head_dim = qkv_size // (num_heads + 2 * num_kv_heads)
    et = t.element_type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        reshaped = ttnn.reshape(
            result=_tt(jit_ctx.ctx, [b, seq, qkv_size], et),
            input=xqkv.mlir_value,
            shape=[b, seq, qkv_size],
        )
        qt = _tt(jit_ctx.ctx, [b, num_heads, seq, head_dim], et)
        kt = _tt(jit_ctx.ctx, [b, num_kv_heads, seq, head_dim], et)
        vt = _tt(jit_ctx.ctx, [b, num_kv_heads, seq, head_dim], et)
        return ttnn.split_query_key_value_and_split_heads(
            qt,
            kt,
            vt,
            reshaped,
            num_heads,
            bool(transpose_k_heads),
            num_kv_heads=num_kv_heads,
        )


def _nlp_create_qkv_heads_decode_handler(
    jit_ctx, xqkv, *, num_heads, num_kv_heads=None, **kwargs
):
    # Decode QKV split: fused [1,1,B,qkv] -> q [1,B,Hq,D], k/v [1,B,Hkv,D].
    # ttnn has a native decode op (no reshape/split workaround needed).
    t = xqkv.mlir_value.type
    shp = [int(d) for d in t.shape]
    batch = shp[-2]
    qkv = shp[-1]
    nkv = num_kv_heads if num_kv_heads is not None else num_heads
    head_dim = qkv // (num_heads + 2 * nkv)
    et = t.element_type
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        qt = _tt(jit_ctx.ctx, [1, batch, num_heads, head_dim], et)
        kt = _tt(jit_ctx.ctx, [1, batch, nkv, head_dim], et)
        vt = _tt(jit_ctx.ctx, [1, batch, nkv, head_dim], et)
        return ttnn.nlp_create_qkv_heads_decode(
            qt, kt, vt, xqkv.mlir_value, num_heads, num_kv_heads=nkv
        )


# --- in-place cache handlers (MemWrite, no result; reads use the cache SSA) --


def _paged_update_cache_handler(
    jit_ctx, cache, input, *, update_idxs_tensor, page_table=None, **kwargs
):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        ttnn.paged_update_cache(
            cache=cache.mlir_value,
            input=input.mlir_value,
            update_index=update_idxs_tensor.mlir_value,
            page_table=(page_table.mlir_value if page_table is not None else None),
        )


def _paged_fused_update_cache_handler(
    jit_ctx, cache1, input1, cache2, input2, *, update_idxs_tensor=None,
    page_table=None, share_cache=None, **kwargs
):
    """``ttnn.experimental.paged_fused_update_cache`` -> TWO ``ttnn.paged_update_cache``.

    DECOMPOSITION, not 1:1. The fused op updates the K and V caches in one
    invocation; the TTNN dialect has PagedUpdateCacheOp but no fused variant, and
    the fused signature is literally (cache1, input1, cache2, input2, ...), so two
    ops express exactly the same writes. The advisor then places the two cache
    updates independently, which for layout purposes is more information, not less
    -- but it is two device ops where the shipped graph runs one, so a
    reconciliation will pair them by position.
    """
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        pt = page_table.mlir_value if page_table is not None else None
        for cache, inp in ((cache1, input1), (cache2, input2)):
            ttnn.paged_update_cache(
                cache=cache.mlir_value,
                input=inp.mlir_value,
                update_index=update_idxs_tensor.mlir_value,
                page_table=pt,
            )


def _paged_fill_cache_handler(jit_ctx, cache, input, page_table, **kwargs):
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        ttnn.paged_fill_cache(
            cache=cache.mlir_value,
            input=input.mlir_value,
            page_table=page_table.mlir_value,
        )


def _make_multi_op(value_fn, jit_ctx):
    def op(*args, **kwargs):
        pa = [_capture(a, jit_ctx) for a in args]
        pk = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        return tuple(TracedTensor(v) for v in value_fn(jit_ctx, *pa, **pk))

    return op


def _make_inplace_op(value_fn, jit_ctx, cache_idx):
    def op(*args, **kwargs):
        raw_cache = args[cache_idx] if cache_idx < len(args) else None
        pa = [_capture(a, jit_ctx) for a in args]
        pk = {k: _capture(v, jit_ctx) for k, v in kwargs.items()}
        value_fn(jit_ctx, *pa, **pk)
        # In-place (MemWrite): the cache SSA value is unchanged; downstream reads
        # of the same cache tensor observe the update. Return the cache proxy so
        # a chained call still gets a TracedTensor.
        cache = pa[cache_idx] if cache_idx < len(pa) else None
        return cache if type(cache) is TracedTensor else None

    return op


def _reduction(op_fn):
    def handler(jit_ctx, x, dim=None, keepdim=False, **kwargs):
        keepdim = kwargs.get("keepdim", kwargs.get("keep_dim", keepdim))
        dim = kwargs.get("dim", kwargs.get("dim_arg", dim))
        shape = [int(d) for d in x.mlir_value.type.shape]
        if dim is None:
            dims = list(range(len(shape)))
        elif isinstance(dim, (list, tuple)):
            dims = [int(d) % len(shape) for d in dim]
        else:
            dims = [int(dim) % len(shape)]
        out = []
        for i, s in enumerate(shape):
            if i in dims:
                if keepdim:
                    out.append(1)
            else:
                out.append(s)
        with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
            rt = _retype(jit_ctx.ctx, x.mlir_value, out)
            return op_fn(
                result=rt, input=x.mlir_value, keep_dim=bool(keepdim), dim_arg=dims
            )

    return handler


def _repeat_handler(jit_ctx, x, repeat_dims=None, **kwargs):
    reps = tuple(
        int(r) for r in (repeat_dims if repeat_dims is not None else kwargs["repeats"])
    )
    shape = [int(d) for d in x.mlir_value.type.shape]
    out = [shape[i] * reps[i] for i in range(len(shape))]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, out)
        repeat_dims_attr = ttnn.ir.ShapeAttr.get(jit_ctx.ctx, reps)
        return ttnn.repeat(result=rt, input=x.mlir_value, repeat_dims=repeat_dims_attr)


_VALUE_HANDLERS = {
    "matmul": _matmul_handler,
    "linear": _linear_handler,
    "sparse_matmul": _sparse_matmul_handler,
    "softplus": _softplus_handler,
    "ones_like": _ones_like_handler,
    "repeat_interleave": _repeat_interleave_handler,
    "pow": _pow_handler,
    "pow_tensor": _pow_handler,
    "reshape": _reshape_handler,
    "typecast": _typecast_handler,
    "softmax": _softmax_handler,
    "rms_norm": _rms_norm_handler,
    "where": _where_handler,
    "slice": _slice_handler,
    "unsqueeze_to_4D": _unsqueeze_to_4d_handler,
    "transpose": _transpose_handler,
    "permute": _permute_handler,
    "concat": _concat_handler,
    "embedding": _embedding_handler,
    "repeat": _repeat_handler,
    # elementwise binary
    "add": _binary(ttnn.add),
    "multiply": _binary(ttnn.multiply),
    "mul": _binary(ttnn.multiply),
    "subtract": _binary(ttnn.subtract),
    "div": _binary(ttnn.divide),
    "divide": _binary(ttnn.divide),
    "maximum": _binary(ttnn.maximum),
    "minimum": _binary(ttnn.minimum),
    "eq": _binary(ttnn.eq),
    "ne": _binary(ttnn.ne),
    "lt": _binary(ttnn.lt),
    "le": _binary(ttnn.le),
    "gt": _binary(ttnn.gt),
    "ge": _binary(ttnn.ge),
    "bitwise_and": _binary(ttnn.bitwise_and),
    "bitwise_or": _binary(ttnn.bitwise_or),
    "bitwise_xor": _binary(ttnn.bitwise_xor),
    # elementwise unary
    "relu": _unary(ttnn.relu),
    "gelu": _unary(ttnn.gelu),
    "silu": _unary(ttnn.silu),
    "sigmoid": _unary(ttnn.sigmoid),
    "hardsigmoid": _unary(ttnn.hardsigmoid),
    "sqrt": _unary(ttnn.sqrt),
    "rsqrt": _unary(ttnn.rsqrt),
    "exp": _unary(ttnn.exp),
    "log": _unary(ttnn.log),
    "neg": _unary(ttnn.neg),
    "tanh": _unary(ttnn.tanh),
    "reciprocal": _unary(ttnn.reciprocal),
    "abs": _unary(ttnn.abs),
    "cos": _unary(ttnn.cos),
    "sin": _unary(ttnn.sin),
    "tan": _unary(ttnn.tan),
    "floor": _unary(ttnn.floor),
    "ceil": _unary(ttnn.ceil),
    "sign": _unary(ttnn.sign),
    "erf": _unary(ttnn.erf),
    "erfc": _unary(ttnn.erfc),
    "logical_and": _binary(ttnn.logical_and),
    "clamp": _clamp_handler,
    "pad": _pad_handler,
    "scatter": _scatter_handler,
    "zeros": _zeros_handler,
    "zeros_like": _zeros_like_handler,
    "arange": _arange_handler,
    "logical_not": _unary(ttnn.logical_not),
    "bitwise_not": _unary(ttnn.bitwise_not),
    # reductions
    "mean": _reduction(ttnn.mean),
    "sum": _reduction(ttnn.sum),
    "max": _reduction(ttnn.max),
    "min": _reduction(ttnn.min),
}

_TOPLEVEL_MULTI = {
    "split": _split_handler,
    "topk": _topk_handler,
}

# name -> positional index of the mutated argument (ttnn returns None).
_TOPLEVEL_INPLACE = {
    "fill_cache": (_fill_cache_handler, 0),
}

# ttnn.experimental.<op> handlers.
_EXPERIMENTAL_VALUE = {
    "rotary_embedding": _rotary_embedding_handler,
    "rotary_embedding_hf": _rotary_embedding_hf_handler,
    "rotary_embedding_llama": _rotary_embedding_llama_handler,
    "nlp_concat_heads": _nlp_concat_heads_handler,
    "nlp_concat_heads_decode": _nlp_concat_heads_decode_handler,
}
_EXPERIMENTAL_MULTI = {
    "nlp_create_qkv_heads": _nlp_create_qkv_heads_handler,
    "nlp_create_qkv_heads_decode": _nlp_create_qkv_heads_decode_handler,
}
# name -> positional index of the mutated cache argument.
_EXPERIMENTAL_INPLACE = {
    "paged_update_cache": (_paged_update_cache_handler, 0),
    "paged_fill_cache": (_paged_fill_cache_handler, 0),
    "paged_fused_update_cache": (_paged_fused_update_cache_handler, 0),
}

# ttnn.transformer.<op> handlers.
_TRANSFORMER_VALUE = {
    "scaled_dot_product_attention": _sdpa_handler,
    "scaled_dot_product_attention_decode": _sdpa_decode_handler,
    "chunked_scaled_dot_product_attention": _chunked_sdpa_handler,
    "paged_scaled_dot_product_attention_decode": _paged_sdpa_decode_handler,
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

# Raw patches: installed WITHOUT _capture, because they need the caller's own
# object identity rather than a TracedTensor proxy.
_RAW = {"copy"}


def _make_raw_copy(jit_ctx):
    """``ttnn.copy(src, dst)`` writes src into dst in place and returns nothing.

    There is no TTNN dialect op for it, so nothing is emitted. The trace-level
    model is to rebind dst's identity to src's value, so a later read of the same
    tensor -- qwen's recurrent/conv state is read on the next token -- observes
    what was written instead of a fresh unrelated placeholder. Requires the raw
    `dst` object, hence _RAW: after _capture, `id(dst)` is a proxy's id.

    Recorded in cache_alias rather than weight_cache: the destination usually
    already HAS a placeholder from an earlier read, and overwriting its
    weight_cache entry orphans that placeholder so _finalize_signature never
    lifts or erases it. A surviving ttir.empty then aborts the pipeline.

    The device write itself is real cost that no op records, so it stays in the
    untraced remainder. That is the honest outcome: the advisor places the ops
    around the state update, not the update.
    """

    def op(src, dst, *args, **kwargs):
        if jit_ctx.cache_alias is None:
            jit_ctx.cache_alias = {}
        if hasattr(src, "mlir_value"):
            jit_ctx.cache_alias[id(dst)] = src.mlir_value
        elif hasattr(src, "shape") and hasattr(src, "dtype"):
            jit_ctx.cache_alias[id(dst)] = _weight_value(src, jit_ctx)
        return None

    return op


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


def _unhandled(name):
    """Stub for an allowlist op the direct-TTNN tracer doesn't emit yet.

    Fails loudly and actionably instead of falling through to a real on-device
    ttnn call (which crashes cryptically on a TracedTensor). This keeps coverage
    gaps visible -- the whole point of the direct-TTNN path is to surface exactly
    which ops still need a handler (or a ttnn dialect op), not to hide them.
    """

    def stub(*args, **kwargs):
        raise NotImplementedError(
            f"ttnn.{name} has no direct-TTNN handler yet (tracer='ttnn'). "
            f"Add one in ttnn_emit_tracer.py, or trace this model with "
            f"tracer='interception' (the TTIR path)."
        )

    return stub


@contextmanager
def patch_ttnn(jit_ctx):
    """Monkeypatch allowlisted ttnn.<op> to build TTNN directly; restore on exit."""
    experimental = getattr(_ttnn_rt, "experimental", None)
    transformer = getattr(_ttnn_rt, "transformer", None)
    originals, exp_originals, tr_originals = {}, {}, {}
    try:
        for name in _PASSTHROUGH_IDENTITY:
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _identity_passthrough)
        for name in _PASSTHROUGH_NONE:
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _none_passthrough)
        for name in _RAW:
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_raw_copy(jit_ctx))
        for name, value_fn in _VALUE_HANDLERS.items():
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_value_op(value_fn, jit_ctx))
        for name, value_fn in _TOPLEVEL_MULTI.items():
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_multi_op(value_fn, jit_ctx))
        for name, (value_fn, idx) in _TOPLEVEL_INPLACE.items():
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_inplace_op(value_fn, jit_ctx, idx))
        if experimental is not None:
            for name, value_fn in _EXPERIMENTAL_VALUE.items():
                exp_originals[name] = getattr(experimental, name, _MISSING)
                setattr(experimental, name, _make_value_op(value_fn, jit_ctx))
            for name, value_fn in _EXPERIMENTAL_MULTI.items():
                exp_originals[name] = getattr(experimental, name, _MISSING)
                setattr(experimental, name, _make_multi_op(value_fn, jit_ctx))
            for name, (value_fn, idx) in _EXPERIMENTAL_INPLACE.items():
                exp_originals[name] = getattr(experimental, name, _MISSING)
                setattr(experimental, name, _make_inplace_op(value_fn, jit_ctx, idx))
        if transformer is not None:
            for name, value_fn in _TRANSFORMER_VALUE.items():
                tr_originals[name] = getattr(transformer, name, _MISSING)
                setattr(transformer, name, _make_value_op(value_fn, jit_ctx))
        # Stub every allowlist op we don't emit yet so it fails loudly instead of
        # falling through to a real on-device ttnn call on a TracedTensor.
        for name in _ALLOWLIST - set(_VALUE_HANDLERS) - set(_TOPLEVEL_MULTI):
            originals.setdefault(name, getattr(_ttnn_rt, name, _MISSING))
            setattr(_ttnn_rt, name, _unhandled(name))
        yield
    finally:
        _restore_patched(_ttnn_rt, originals)
        if experimental is not None:
            _restore_patched(experimental, exp_originals)
        if transformer is not None:
            _restore_patched(transformer, tr_originals)


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
