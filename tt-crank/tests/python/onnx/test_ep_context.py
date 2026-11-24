# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""EPContext: compile a model to a precompiled .onnx, then run it without recompiling."""

import numpy as np
import pytest

import tt_crank.onnx as tt_onnx


def _gemm_relu() -> bytes:
    return tt_onnx.model(
        [
            tt_onnx.node("Gemm", ["x", "w"], ["y"], transB=1),
            tt_onnx.node("Relu", ["y"], ["z"]),
        ],
        [tt_onnx.input("x", [4, 16])],
        [tt_onnx.output("z", [4, 8])],
        [tt_onnx.constant("w", tt_onnx.randn(8, 16) / 4)],
    )


def _gemm_relu_reshape() -> bytes:
    # Reshape's shape is a build-time constant, not a parameter; a leaked dead
    # parameter would desync the context blob.
    return tt_onnx.model(
        [
            tt_onnx.node("Gemm", ["x", "w", "b"], ["y"], transB=1),
            tt_onnx.node("Relu", ["y"], ["r"]),
            tt_onnx.node("Reshape", ["r", "shape"], ["z"]),
        ],
        [tt_onnx.input("x", [4, 8])],
        [tt_onnx.output("z", [2, 2, 16])],
        [
            tt_onnx.constant("w", tt_onnx.randn(16, 8) / 4),
            tt_onnx.constant("b", tt_onnx.randn(16) / 4),
            tt_onnx.constant("shape", np.array([2, 2, 16], dtype=np.int64)),
        ],
    )


@pytest.mark.parametrize(
    "build,in_shape",
    [(_gemm_relu, (4, 16)), (_gemm_relu_reshape, (4, 8))],
    ids=["gemm_relu", "gemm_relu_reshape"],
)
def test_ep_context_round_trip(build, in_shape, tmp_path) -> None:
    model = build()
    inputs = {"x": tt_onnx.randn(*in_shape)}
    want = tt_onnx.cpu_golden(model, inputs)

    ctx_path = tmp_path / "model_ctx.onnx"
    tt_onnx.session(model, ep_context=ctx_path)  # compiling writes the context model
    assert ctx_path.exists()

    got = tt_onnx.run(tt_onnx.session(ctx_path), inputs)
    np.testing.assert_allclose(got[0], want[0], atol=2e-2, rtol=2e-2)


def test_ep_context_default_path(tmp_path) -> None:
    # Without an explicit file, ORT writes <model>_ctx.onnx next to a model loaded from disk.
    path = tmp_path / "model.onnx"
    path.write_bytes(_gemm_relu())
    tt_onnx.session(path, ep_context=True)
    assert (tmp_path / "model_ctx.onnx").exists()
