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
