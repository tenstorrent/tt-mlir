# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device OrtValues: the EP's memory identity (allocator + data transfer).

Creation must go through the EpDevice's memory_info (the by-device-string
lookup in ortvalue_from_numpy('npu', ...) doesn't resolve plugin shared
allocators in ORT 1.29 — see microsoft/onnxruntime issue 32164).
"""

import numpy as np
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper
from onnxruntime.capi import _pybind_state as C

from ort_ep_utils import EP_OPSET_VERSION, session_on_tt, tt_device


def _model_bytes() -> bytes:
    rng = np.random.default_rng(0)
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "w"], ["g"], transB=1),
            helper.make_node("Relu", ["g"], ["y"]),
        ],
        "device_values",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 128])],
        initializer=[
            helper.make_tensor(
                "w",
                TensorProto.FLOAT,
                [128, 64],
                rng.standard_normal((128, 64), dtype=np.float32).flatten(),
            )
        ],
    )
    return helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", EP_OPSET_VERSION)]
    ).SerializeToString()


@pytest.fixture()
def session() -> ort.InferenceSession:
    return session_on_tt(_model_bytes())


def test_device_ortvalue_roundtrip() -> None:
    # Allocate on the TT device (box allocator) and copy through the TT data
    # transfer in both directions.
    mi = tt_device().memory_info(C.OrtDeviceMemoryType.DEFAULT)
    x = np.random.default_rng(1).standard_normal((32, 64), dtype=np.float32)
    dev = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 64], np.float32, memory_info=mi
    )
    ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(x)], [dev])
    back = ort.OrtValue.ortvalue_from_numpy(np.zeros_like(x))
    ort.copy_tensors([dev], [back])
    np.testing.assert_array_equal(x, back.numpy())


def test_iobinding_device_input_reused(session: ort.InferenceSession) -> None:
    mi = tt_device().memory_info(C.OrtDeviceMemoryType.DEFAULT)
    x = np.random.default_rng(1).standard_normal((32, 64), dtype=np.float32)
    (y_plain,) = session.run(None, {"x": x})

    x_dev = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 64], np.float32, memory_info=mi
    )
    ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(x)], [x_dev])

    io = session.io_binding()
    io.bind_ortvalue_input("x", x_dev)
    io.bind_output("y", "cpu")
    # Two runs: the second reuses the device tensor the first bind uploaded
    # (the box handle is written back, torch-storage style).
    for _ in range(2):
        session.run_with_iobinding(io)
        (y,) = io.copy_outputs_to_cpu()
        np.testing.assert_allclose(y_plain, y, atol=1e-5)


def test_iobinding_device_output(session: ort.InferenceSession) -> None:
    mi = tt_device().memory_info(C.OrtDeviceMemoryType.DEFAULT)
    x = np.random.default_rng(2).standard_normal((32, 64), dtype=np.float32)
    (y_plain,) = session.run(None, {"x": x})

    x_dev = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 64], np.float32, memory_info=mi
    )
    ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(x)], [x_dev])
    y_dev = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 128], np.float32, memory_info=mi
    )

    io = session.io_binding()
    io.bind_ortvalue_input("x", x_dev)
    io.bind_ortvalue_output("y", y_dev)
    session.run_with_iobinding(io)

    # The output lives on device until explicitly copied back.
    y_back = ort.OrtValue.ortvalue_from_numpy(np.zeros((32, 128), dtype=np.float32))
    ort.copy_tensors([y_dev], [y_back])
    np.testing.assert_allclose(y_plain, y_back.numpy(), atol=1e-5)


def test_device_to_device_copy() -> None:
    # TT->TT copies into a fresh (empty-box) destination, both while the
    # source is still host-staged and after a run has uploaded it.
    mi = tt_device().memory_info(C.OrtDeviceMemoryType.DEFAULT)
    x = np.random.default_rng(3).standard_normal((32, 32), dtype=np.float32)

    src = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 32], np.float32, memory_info=mi
    )
    ort.copy_tensors([ort.OrtValue.ortvalue_from_numpy(x)], [src])  # host-staged
    dst = ort.OrtValue.ortvalue_from_shape_and_type(
        [32, 32], np.float32, memory_info=mi
    )
    ort.copy_tensors([src], [dst])

    back = ort.OrtValue.ortvalue_from_numpy(np.zeros_like(x))
    ort.copy_tensors([dst], [back])
    np.testing.assert_array_equal(x, back.numpy())
