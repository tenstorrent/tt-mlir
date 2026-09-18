# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device OrtValues: the EP's allocator and data transfer.

Device values are created through the EpDevice's memory_info; the by-name path
(ortvalue_from_numpy('npu', ...)) doesn't resolve plugin allocators in ORT 1.29
(microsoft/onnxruntime#32164).
"""

import numpy as np
import onnxruntime as ort
import pytest

import tt_crank.onnx as tt_onnx

_GEMM_RELU = tt_onnx.make_model(
    [
        tt_onnx.make_node("Gemm", ["x", "w"], ["g"], transB=1),
        tt_onnx.make_node("Relu", ["g"], ["y"]),
    ],
    [tt_onnx.vi("x", [32, 64])],
    [tt_onnx.vi("y", [32, 128])],
    [tt_onnx.make_tensor("w", tt_onnx.randn(128, 64))],
)


@pytest.fixture()
def session() -> ort.InferenceSession:
    return tt_onnx.session(_GEMM_RELU)


def test_device_ortvalue_roundtrip() -> None:
    x = tt_onnx.randn(32, 64)
    np.testing.assert_array_equal(x, tt_onnx.to_host(tt_onnx.to_device(x)))


def test_iobinding_device_input_reused(session: ort.InferenceSession) -> None:
    x = tt_onnx.randn(32, 64)
    (y_plain,) = session.run(None, {"x": x})

    io = session.io_binding()
    io.bind_ortvalue_input("x", tt_onnx.to_device(x))
    io.bind_output("y", "cpu")
    # The second run reuses the device tensor the first one uploaded.
    for _ in range(2):
        session.run_with_iobinding(io)
        (y,) = io.copy_outputs_to_cpu()
        np.testing.assert_allclose(y_plain, y, atol=1e-5)


def test_iobinding_device_output(session: ort.InferenceSession) -> None:
    x = tt_onnx.randn(32, 64)
    (y_plain,) = session.run(None, {"x": x})

    y_dev = tt_onnx.empty((32, 128), np.float32)
    io = session.io_binding()
    io.bind_ortvalue_input("x", tt_onnx.to_device(x))
    io.bind_ortvalue_output("y", y_dev)
    session.run_with_iobinding(io)
    np.testing.assert_allclose(y_plain, tt_onnx.to_host(y_dev), atol=1e-5)


def test_device_to_device_copy() -> None:
    x = tt_onnx.randn(32, 32)
    dst = tt_onnx.empty(x.shape, np.float32)
    ort.copy_tensors([tt_onnx.to_device(x)], [dst])
    np.testing.assert_array_equal(x, tt_onnx.to_host(dst))
