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
    assert "ttir.empty" in ir  # captured weight
    # exactly one declared parameter (x); the weight is not a param:
    assert len(module.body.operations[0].arguments) == 1
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
        h = ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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

    with InsertionPoint(scope.func_bb), Location.unknown(scope.ctx):
        et = x.mlir_value.type.element_type
        qt = RankedTensorType.get([1, 32, 32, 128], et)
        kt = RankedTensorType.get([1, 8, 32, 128], et)
        vt = RankedTensorType.get([1, 8, 32, 128], et)
        results = ttir.split_query_key_value_and_split_heads(
            qt, kt, vt, x.mlir_value, 32, False, num_kv_heads=8
        )
    wrapped = _wrap_results(results)
    assert isinstance(wrapped, tuple) and len(wrapped) == 3
    assert all(hasattr(w, "mlir_value") for w in wrapped)
    assert wrapped[0].shape == (1, 32, 32, 128)
    assert wrapped[1].shape == (1, 8, 32, 128)


def test_nlp_create_qkv_heads_multi_output():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((1, 1, 32, 6144), _ttnn.bfloat16)])  # 6144 = 128*(32+2*8)
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
        out = _ttnn.transformer.scaled_dot_product_attention(q, kdummy, vdummy, is_causal=True)
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


def test_chunked_sdpa_output_matches_query_shape():
    import ttnn as _ttnn
    from ttnn_jit.ttmlir.ir import InsertionPoint, Location
    from ttnn_jit.ttmlir.dialects import func as _func

    scope = build_trace_scope("f", [((1, 32, 32, 128), _ttnn.bfloat16)])
    q = scope.traced_args[0]
    kdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    vdummy = _DummyTensor((1, 8, 32, 128), _ttnn.bfloat16)
    page_table = _DummyTensor((1, 4), _ttnn.uint32)  # captured but unused by the SDPA mapping
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
