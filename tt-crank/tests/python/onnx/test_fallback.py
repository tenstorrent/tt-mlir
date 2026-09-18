# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Node support: unclaimed nodes fall back to the CPU EP; a claimed node the builder can't lower fails session creation."""

import numpy as np
import pytest

import tt_onnx

_TANH = tt_onnx.make_model(
    [tt_onnx.make_node("Tanh", ["x"], ["y"])],
    [tt_onnx.vi("x", [16, 16])],
    [tt_onnx.vi("y", [16, 16])],
)


def test_unsupported_op_falls_back_to_cpu() -> None:
    # Tanh is not in the EP's op set: the whole graph runs on the CPU EP.
    inputs = {"x": tt_onnx.randn(16, 16)}
    got = tt_onnx.session_on_tt(_TANH, allow_cpu_fallback=True).run(None, inputs)
    np.testing.assert_allclose(got[0], tt_onnx.cpu_golden(_TANH, inputs)[0], atol=1e-5)


def test_unsupported_op_without_fallback_fails_session_creation() -> None:
    with pytest.raises(
        Exception, match="fallback to CPU EP has been explicitly disabled"
    ):
        tt_onnx.session_on_tt(_TANH)


def test_zero_sized_dim_falls_back_to_cpu() -> None:
    # Legal ONNX the compiler can't take: declined, then computed on CPU.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("Add", ["a", "b"], ["y"])],
        [tt_onnx.vi("a", [0, 32]), tt_onnx.vi("b", [0, 32])],
        [tt_onnx.vi("y", [0, 32])],
    )
    inputs = {"a": tt_onnx.randn(0, 32), "b": tt_onnx.randn(0, 32)}
    (got,) = tt_onnx.session_on_tt(model, allow_cpu_fallback=True).run(None, inputs)
    assert got.shape == (0, 32)


def test_batch_norm_rank6_fails_session_creation() -> None:
    # Valid ONNX, but ttir.batch_norm_inference caps rank at 2..5; the node is claimed and the build fails.
    model = tt_onnx.make_model(
        [tt_onnx.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"])],
        [tt_onnx.vi("x", [2, 4, 2, 2, 8, 8])],
        [tt_onnx.vi("y", None)],
        [
            tt_onnx.make_tensor("s", tt_onnx.randn(4)),
            tt_onnx.make_tensor("b", tt_onnx.randn(4)),
            tt_onnx.make_tensor("m", tt_onnx.randn(4)),
            tt_onnx.make_tensor("v", np.abs(tt_onnx.randn(4)) + 0.5),
        ],
    )
    with pytest.raises(Exception, match="outside the supported 2..5"):
        tt_onnx.session_on_tt(model, allow_cpu_fallback=True)
