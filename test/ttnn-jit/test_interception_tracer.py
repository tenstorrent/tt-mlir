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
