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
in-place paged KV-cache ops. The full Llama decoder sweeps in both phases. Not
yet ported: the long-tail allowlisted ops the TTIR tracer covers via
BaseOpHandler (reductions, the rest of the tm/unary/binary set) and the MoE ops
(topk/scatter/sparse_matmul); trace models needing those through the default
TTIR path (pipeline="scoped").
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
    reps = [
        int(r) for r in (repeat_dims if repeat_dims is not None else kwargs["repeats"])
    ]
    shape = [int(d) for d in x.mlir_value.type.shape]
    out = [shape[i] * reps[i] for i in range(len(shape))]
    with InsertionPoint(jit_ctx.func_bb), Location.unknown(jit_ctx.ctx):
        rt = _retype(jit_ctx.ctx, x.mlir_value, out)
        return ttnn.repeat(result=rt, input=x.mlir_value, repeat_dims=reps)


_VALUE_HANDLERS = {
    "matmul": _matmul_handler,
    "linear": _linear_handler,
    "reshape": _reshape_handler,
    "typecast": _typecast_handler,
    "softmax": _softmax_handler,
    "rms_norm": _rms_norm_handler,
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
    "logical_not": _unary(ttnn.logical_not),
    "bitwise_not": _unary(ttnn.bitwise_not),
    # reductions
    "mean": _reduction(ttnn.mean),
    "sum": _reduction(ttnn.sum),
    "max": _reduction(ttnn.max),
    "min": _reduction(ttnn.min),
}

# ttnn.experimental.<op> handlers.
_EXPERIMENTAL_VALUE = {
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
}

# ttnn.transformer.<op> handlers.
_TRANSFORMER_VALUE = {
    "scaled_dot_product_attention": _sdpa_handler,
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
        for name, value_fn in _VALUE_HANDLERS.items():
            originals[name] = getattr(_ttnn_rt, name, _MISSING)
            setattr(_ttnn_rt, name, _make_value_op(value_fn, jit_ctx))
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
        for name in _ALLOWLIST - set(_VALUE_HANDLERS):
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
