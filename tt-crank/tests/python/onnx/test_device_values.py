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
from onnx import helper

from ort_ep_utils import (
    randn,
    from_tt,
    make_tensor,
    make_model,
    session_on_tt,
    to_tt,
    tt_empty,
    vi,
)

_GEMM_RELU = make_model(
    [
        helper.make_node("Gemm", ["x", "w"], ["g"], transB=1),
        helper.make_node("Relu", ["g"], ["y"]),
    ],
    [vi("x", [32, 64])],
    [vi("y", [32, 128])],
    [make_tensor("w", randn(128, 64))],
)


@pytest.fixture()
def session() -> ort.InferenceSession:
    return session_on_tt(_GEMM_RELU)


def test_device_ortvalue_roundtrip() -> None:
    x = randn(32, 64)
    np.testing.assert_array_equal(x, from_tt(to_tt(x)))


def test_iobinding_device_input_reused(session: ort.InferenceSession) -> None:
    x = randn(32, 64)
    (y_plain,) = session.run(None, {"x": x})

    io = session.io_binding()
    io.bind_ortvalue_input("x", to_tt(x))
    io.bind_output("y", "cpu")
    # The second run reuses the device tensor the first one uploaded.
    for _ in range(2):
        session.run_with_iobinding(io)
        (y,) = io.copy_outputs_to_cpu()
        np.testing.assert_allclose(y_plain, y, atol=1e-5)


def test_iobinding_device_output(session: ort.InferenceSession) -> None:
    x = randn(32, 64)
    (y_plain,) = session.run(None, {"x": x})

    y_dev = tt_empty((32, 128), np.float32)
    io = session.io_binding()
    io.bind_ortvalue_input("x", to_tt(x))
    io.bind_ortvalue_output("y", y_dev)
    session.run_with_iobinding(io)
    np.testing.assert_allclose(y_plain, from_tt(y_dev), atol=1e-5)


def test_device_to_device_copy() -> None:
    x = randn(32, 32)
    dst = tt_empty(x.shape, np.float32)
    ort.copy_tensors([to_tt(x)], [dst])
    np.testing.assert_array_equal(x, from_tt(dst))
