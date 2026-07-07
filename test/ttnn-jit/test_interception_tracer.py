# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import pytest

from ttnn_jit._src.interception_tracer import TracedTensor, build_trace_scope


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
