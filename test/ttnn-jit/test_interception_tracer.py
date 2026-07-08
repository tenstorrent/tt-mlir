# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest

from ttnn_jit._src.interception_tracer import (
    TracedTensor,
    build_trace_scope,
    patch_ttnn,
)


class _DummyTensor:
    """Stand-in for a ttnn.Tensor in device-free tests (metadata only)."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def test_traced_tensor_exposes_shape_and_dtype():
    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    x = scope.traced_args[0]
    assert isinstance(x, TracedTensor)
    assert x.shape == (256, 512)
    assert x.dtype == ttnn.bfloat16


def test_traced_tensor_layout_is_unknown():
    scope = build_trace_scope("f", [((32, 32), ttnn.bfloat16)])
    x = scope.traced_args[0]
    with pytest.raises(NotImplementedError):
        x.memory_config()
    with pytest.raises(NotImplementedError):
        _ = x.layout


def test_patch_ttnn_restores_originals_including_on_exception():
    original_matmul = ttnn.matmul
    # Capture submodule originals BEFORE any patching so we can confirm
    # patch_ttnn's finally block restores nested (ttnn.experimental /
    # ttnn.transformer) attrs too, not just top-level ttnn ones.
    has_sdpa = hasattr(ttnn, "transformer") and hasattr(
        ttnn.transformer, "scaled_dot_product_attention"
    )
    if has_sdpa:
        original_sdpa = ttnn.transformer.scaled_dot_product_attention
    has_concat_heads = hasattr(ttnn, "experimental") and hasattr(
        ttnn.experimental, "nlp_concat_heads"
    )
    if has_concat_heads:
        original_concat_heads = ttnn.experimental.nlp_concat_heads
    has_qkv_heads = hasattr(ttnn, "experimental") and hasattr(
        ttnn.experimental, "nlp_create_qkv_heads"
    )
    if has_qkv_heads:
        original_qkv_heads = ttnn.experimental.nlp_create_qkv_heads

    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    with patch_ttnn(scope.jit_ctx):
        assert ttnn.matmul is not original_matmul
    assert ttnn.matmul is original_matmul

    try:
        with patch_ttnn(scope.jit_ctx):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert ttnn.matmul is original_matmul
    if has_sdpa:
        assert ttnn.transformer.scaled_dot_product_attention is original_sdpa
    if has_concat_heads:
        assert ttnn.experimental.nlp_concat_heads is original_concat_heads
    if has_qkv_heads:
        assert ttnn.experimental.nlp_create_qkv_heads is original_qkv_heads


def test_patched_op_builds_ttir_and_captures_weight():
    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    x = scope.traced_args[0]
    w = _DummyTensor((512, 512), ttnn.bfloat16)  # captured weight (not a proxy)
    with patch_ttnn(scope.jit_ctx):
        out = ttnn.matmul(x, w)
    assert isinstance(out, TracedTensor)
    assert out.shape == (256, 512)
    ir = str(scope.module)
    assert "ttir.matmul" in ir
    assert "ttir.empty" in ir  # weight materialized


def test_trace_intercepted_cross_function_shape_and_weight_capture():
    from ttnn_jit._src.interception_tracer import trace_intercepted

    # Weight captured from the enclosing scope (not a declared param).
    W = _DummyTensor((512, 512), ttnn.bfloat16)

    def _linear(a, b):
        return ttnn.matmul(a, b)  # called INSIDE a helper -> cross-fn interception

    def mlp(x):
        h = _linear(x, W)  # W captured -> weight capture
        h = ttnn.relu(h)
        rows, cols = h.shape[0], h.shape[1]  # reads proxy .shape
        h = ttnn.reshape(h, shape=(rows, cols))  # shape kwarg (see Global Constraints)
        return ttnn.add(h, h)

    module, out_type = trace_intercepted(mlp, _DummyTensor((256, 512), ttnn.bfloat16))

    ir = str(module)
    assert "ttir.matmul" in ir  # emitted from the helper
    assert "ttir.relu" in ir
    assert "ttir.reshape" in ir
    assert "ttir.add" in ir
    # The captured weight W is lifted to a function argument (tagged as a
    # parameter), not left as a ttir.empty placeholder, so the optimizer models
    # it as a DRAM-resident weight rather than a freshly-allocated intermediate.
    assert "ttir.empty" not in ir
    # Two args now: the declared activation x (input) + the captured weight W
    # (parameter).
    assert len(module.body.operations[0].arguments) == 2
    assert "#ttcore.argument_type<input>" in ir
    assert "#ttcore.argument_type<parameter>" in ir
    assert tuple(int(d) for d in out_type.shape) == (256, 512)


def test_passthrough_to_memory_config_is_identity_and_emits_no_op():
    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    x = scope.traced_args[0]
    with patch_ttnn(scope.jit_ctx):
        out = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        also = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        dealloc = ttnn.deallocate(x)
    assert out is x  # identity: same proxy back
    assert also is x
    assert dealloc is None
    assert "to_memory_config" not in str(scope.module)
    assert "ttir.empty" not in str(scope.module)  # x not re-materialized


def test_capture_walks_kwargs_and_ignores_non_tensors():
    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    x = scope.traced_args[0]
    w = _DummyTensor((256, 512), ttnn.bfloat16)  # tensor passed by KEYWORD to clamp
    with patch_ttnn(scope.jit_ctx):
        # clamp with tensor-valued min kwarg (exercises kwargs capture)
        out = ttnn.clamp(x, min=w)
        # Also test that non-tensor config kwarg passes through without materializing
        out2 = ttnn.multiply(x, x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert isinstance(out, TracedTensor)
    ir = str(scope.module)
    assert "ttir.empty" in ir  # w kwarg tensor materialized
    assert out2 is not None  # memory_config kwarg did not crash


def test_common_op_handlers_build_ttir():
    scope = build_trace_scope("f", [((256, 512), ttnn.bfloat16)])
    x = scope.traced_args[0]
    w = _DummyTensor((512, 1024), ttnn.bfloat16)
    norm_w = _DummyTensor((512,), ttnn.bfloat16)
    with patch_ttnn(scope.jit_ctx):
        h = ttnn.linear(
            x, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert h.shape == (256, 1024)
        g = ttnn.mul(x, x)  # alias for multiply
        assert g.shape == (256, 512)
        c = ttnn.typecast(x, ttnn.bfloat16)
        assert c.shape == (256, 512)
        n = ttnn.rms_norm(x, epsilon=1e-5, weight=norm_w)
        assert n.shape == (256, 512)
    ir = str(scope.module)
    assert "ttir.linear" in ir
    assert "ttir.multiply" in ir
    assert "ttir.typecast" in ir
    assert "ttir.rms_norm" in ir


def test_multi_output_returns_tuple_of_proxies():
    from ttnn_jit._src.interception_tracer import _wrap_results

    scope = build_trace_scope("f", [((1, 1, 32, 6144), ttnn.bfloat16)])
    x = scope.traced_args[0]
    # Emit a real 3-result op to get 3 MLIR values, then wrap them.
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location, RankedTensorType
    from ttnn_jit.ttmlir.dialects import ttir
    from ttnn_jit.ttmlir.dialects import func as _func

    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        et = x.mlir_value.type.element_type
        # The SplitQueryKeyValueAndSplitHeadsOp verifier requires a rank-3
        # [batch, seq, qkv_size] input; reshape the rank-4 fused-qkv block arg
        # down first, matching what _nlp_create_qkv_heads_handler does for the
        # real op.
        reshaped_type = RankedTensorType.get([1, 32, 6144], et)
        reshaped = ttir.reshape(reshaped_type, x.mlir_value, [1, 32, 6144])
        qt = RankedTensorType.get([1, 32, 32, 128], et)
        kt = RankedTensorType.get([1, 8, 32, 128], et)
        vt = RankedTensorType.get([1, 8, 32, 128], et)
        results = ttir.split_query_key_value_and_split_heads(
            qt, kt, vt, reshaped, 32, False, num_kv_heads=8
        )
    wrapped = _wrap_results(results)
    assert isinstance(wrapped, tuple) and len(wrapped) == 3
    assert all(hasattr(w, "mlir_value") for w in wrapped)
    assert wrapped[0].shape == (1, 32, 32, 128)
    assert wrapped[1].shape == (1, 8, 32, 128)
    # build_trace_scope's func body has no terminator (this test doesn't run
    # the full trace_intercepted finalize step) -- add one so operation.verify()
    # is a real oracle over the reshape + split ops themselves. x's type
    # matches the func's pre-declared result type, so returning it unchanged
    # keeps the signature consistent.
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([x.mlir_value])
    scope.module.operation.verify()


def test_nlp_create_qkv_heads_multi_output():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope(
        "f", [((1, 1, 32, 6144), _ttnn.bfloat16)]
    )  # 6144 = 128*(32+2*8)
    x = scope.traced_args[0]
    with patch_ttnn(scope.jit_ctx):
        q, k, v = _ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=32, num_kv_heads=8, transpose_k_heads=False
        )
    assert q.shape == (1, 32, 32, 128)
    assert k.shape == (1, 8, 32, 128)
    assert v.shape == (1, 8, 32, 128)
    assert "split_query_key_value_and_split_heads" in str(scope.module)
    # build_trace_scope's func body has no terminator (this test doesn't run
    # the full trace_intercepted finalize step) -- add one so operation.verify()
    # exercises the split op itself rather than failing on a missing return.
    # x's type matches the func's pre-declared result type, so returning it
    # unchanged keeps the signature consistent.
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([x.mlir_value])
    scope.module.operation.verify()


def test_sdpa_output_matches_query_shape():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((1, 32, 32, 128), _ttnn.bfloat16)])
    q = scope.traced_args[0]
    kdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    vdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    with patch_ttnn(scope.jit_ctx):
        out = _ttnn.transformer.scaled_dot_product_attention(
            q, kdummy, vdummy, is_causal=True
        )
    assert out.shape == (1, 32, 32, 128)
    assert "scaled_dot_product_attention" in str(scope.module)
    # build_trace_scope's func body has no terminator (this test doesn't run
    # the full trace_intercepted finalize step) -- add one so operation.verify()
    # exercises the SDPA op itself. out's type matches the func's pre-declared
    # result type (query type == result type per the SDPA verifier), so
    # returning it keeps the signature consistent.
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([out.mlir_value])
    scope.module.operation.verify()


def test_concat_heads_merges():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((1, 32, 32, 128), _ttnn.bfloat16)])
    x = scope.traced_args[0]
    with patch_ttnn(scope.jit_ctx):
        out = _ttnn.experimental.nlp_concat_heads(x)
    assert "concatenate_heads" in str(scope.module)
    assert out.shape == (1, 32, 4096)  # [b, seq, num_heads * head_dim]
    assert out.shape[0] == 1  # batch preserved
    # build_trace_scope's func body has no terminator, and its pre-declared
    # result type matches x (rank 4), not out (rank 3, per the
    # ConcatenateHeadsOp verifier's [b, seq, num_heads*head_dim] convention)
    # -- return x unchanged so operation.verify() exercises the
    # concatenate_heads op itself rather than failing on a return-type
    # mismatch.
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([x.mlir_value])
    scope.module.operation.verify()


def test_paged_fill_cache_returns_cache_shape():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((4, 8, 32, 128), _ttnn.bfloat16)])
    cache = scope.traced_args[0]
    k_fill = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)  # single-user slice
    page_table = _DummyTensor((1, 4), _ttnn.uint32)
    with patch_ttnn(scope.jit_ctx):
        # Real call sites pass a scalar batch_idx (not batch_idx_tensor); the
        # handler ignores it via **kwargs -- the TTIR op only models the
        # optional batch_idx_tensor operand.
        out = _ttnn.experimental.paged_fill_cache(
            cache, k_fill, page_table, batch_idx=0
        )
    assert out.shape == (4, 8, 32, 128)  # result = cache shape
    assert "paged_fill_cache" in str(scope.module)
    # out's type matches cache's type, which matches the func's pre-declared
    # result type (input_types[0]) -- return it directly.
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([out.mlir_value])
    scope.module.operation.verify()


def test_chunked_sdpa_output_matches_query_shape():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((1, 32, 32, 128), _ttnn.bfloat16)])
    q = scope.traced_args[0]
    kdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    vdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    page_table = _DummyTensor(
        (1, 4), _ttnn.uint32
    )  # captured but unused by the SDPA mapping
    with patch_ttnn(scope.jit_ctx):
        out = _ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q,
            input_tensor_k=kdummy,
            input_tensor_v=vdummy,
            page_table_tensor=page_table,
        )
    assert out.shape == (1, 32, 32, 128)
    assert "scaled_dot_product_attention" in str(scope.module)
    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        _func.ReturnOp([out.mlir_value])
    scope.module.operation.verify()


def test_synthetic_attention_chain_traces():
    import ttnn as _ttnn
    from ttnn_jit._src.interception_tracer import trace_intercepted

    def attn(xqkv):
        q, k, v = _ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=32, num_kv_heads=8, transpose_k_heads=False
        )
        # (RoPE omitted — Plan C.) Repeat kv to match q heads is handled inside sdpa;
        # here we pass q,k,v straight to sdpa for the op-chain smoke.
        attn_out = _ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        return _ttnn.experimental.nlp_concat_heads(attn_out)

    module, out_type = trace_intercepted(
        attn, _DummyTensor((1, 1, 32, 6144), _ttnn.bfloat16)
    )
    ir = str(module)
    assert "split_query_key_value_and_split_heads" in ir
    assert "scaled_dot_product_attention" in ir
    assert "concatenate_heads" in ir
    module.operation.verify()


def test_rotary_embedding_llama_handler():
    import ttnn as _ttnn

    scope = build_trace_scope("f", [((1, 32, 32, 128), _ttnn.bfloat16)])
    x = scope.traced_args[0]
    cos = _DummyTensor((1, 1, 32, 128), _ttnn.bfloat16)
    sin = _DummyTensor((1, 1, 32, 128), _ttnn.bfloat16)
    tm = _DummyTensor((1, 1, 32, 32), _ttnn.bfloat16)
    with patch_ttnn(scope.jit_ctx):
        out = _ttnn.experimental.rotary_embedding_llama(
            x, cos, sin, tm, is_decode_mode=False
        )
    assert out.shape == (1, 32, 32, 128)
    assert "rotary_embedding_llama" in str(scope.module)
