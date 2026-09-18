# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the tt-kurbla ONNX Runtime plugin-EP tests."""

import os
import pathlib

import numpy as np
import onnxruntime as ort
from onnx import helper, shape_inference

TT_MLIR_ROOT = pathlib.Path(__file__).resolve().parents[4]
EP_NAME = "TTKurblaExecutionProvider"
REGISTRATION_NAME = "tt_kurbla"
EP_OPSET_VERSION = 22


def ep_library_path() -> pathlib.Path:
    """The built plugin library (BUILD_DIR overrides <tt-mlir>/build)."""
    build_dir = pathlib.Path(os.environ.get("BUILD_DIR", TT_MLIR_ROOT / "build"))
    return build_dir / "tt-crank" / "src" / "onnx" / "libtt_crank_ort.so"


def tt_device():
    return next(d for d in ort.get_ep_devices() if d.ep_name == EP_NAME)


def session_opts():
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _as_model(model: str | pathlib.Path | bytes) -> str | bytes:
    # ORT's InferenceSession takes a path string or the model bytes; a Path
    # must be stringified.
    return str(model) if isinstance(model, pathlib.Path) else model


def session_on_tt(
    model: str | pathlib.Path | bytes,
    allow_cpu_fallback: bool = False,
    compile_options: dict[str, str] | None = None,
    free_dims: dict[str, int] | None = None,
    ctx_enabled: bool = False,
    ctx_file_path: str | pathlib.Path = None,
) -> ort.InferenceSession:
    """A session with the tt EP.
    Model can be loaded from disk (.onnx file) when ctx_enabled or from model bytes (e.g. torch.onnx.export).
    `free_dims` pins symbolic input dimensions by name (e.g. a dynamic batch "N")."""
    options = session_opts()
    options.add_provider_for_devices([tt_device()], compile_options or {})

    if not allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    if ctx_enabled:
        options.add_session_config_entry("ep.context_enable", "1")
        if ctx_file_path != None:
            options.add_session_config_entry("ep.context_file_path", str(ctx_file_path))

    for name, value in (free_dims or {}).items():
        options.add_free_dimension_override_by_name(name, value)

    return ort.InferenceSession(_as_model(model), sess_options=options)


def cpu_golden(
    model: str | pathlib.Path | bytes, feeds: dict[str, np.ndarray]
) -> list[np.ndarray]:
    session = ort.InferenceSession(
        _as_model(model), session_opts(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, feeds)


# ---- model-building helpers --------------------------------------------------

_RNG = np.random.default_rng(0)


def _vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def _model(nodes, inputs, outputs, initializers=(), opset=EP_OPSET_VERSION) -> bytes:
    graph = helper.make_graph(
        list(nodes),
        "ep_op_test",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return shape_inference.infer_shapes(model).SerializeToString()


def _f32(*shape: int) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(np.float32)


def _assert_tt_matches_cpu(model: bytes, feeds: dict, atol: float, rtol: float) -> None:
    got = session_on_tt(model).run(None, feeds)
    want = cpu_golden(model, feeds)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=atol, rtol=rtol)


# ---- device tensor helpers ---------------------------------------------------
# numpy-backed OrtValues of any EP dtype, moved to/from the TT device (no torch). bf16 has no native numpy
# scalar, so it uses ml_dtypes.bfloat16 and crosses via ortvalue_from_numpy_with_onnx_type (ORT >= 1.20).


def tt_memory_info():
    """DEFAULT memory info for the TT EP device — used to allocate device OrtValues."""
    from onnxruntime.capi import _pybind_state as C

    return tt_device().memory_info(C.OrtDeviceMemoryType.DEFAULT)


def _bf16_onnx_type(dtype) -> int | None:
    """int ONNX element type for a numpy dtype ORT's plain-numpy path can't map
    (i.e. ml_dtypes.bfloat16); None for dtypes numpy/ORT handle natively."""
    import ml_dtypes
    import onnx

    return (
        int(onnx.TensorProto.BFLOAT16)
        if np.dtype(dtype) == np.dtype(ml_dtypes.bfloat16)
        else None
    )


def _numpy_dtype(ov: ort.OrtValue):
    """numpy dtype for an OrtValue's element type (ml_dtypes.bfloat16 for bf16)."""
    import ml_dtypes

    name = ov.data_type()[len("tensor(") : -1]  # "tensor(bfloat16)" -> "bfloat16"
    return {
        "float": np.float32,
        "double": np.float64,
        "float16": np.float16,
        "bfloat16": ml_dtypes.bfloat16,
        "int64": np.int64,
        "int32": np.int32,
        "bool": np.bool_,
    }[name]


def host_ortvalue(arr: np.ndarray) -> ort.OrtValue:
    """A host OrtValue aliasing numpy array `arr` (any EP dtype, incl.
    ml_dtypes.bfloat16). bf16 goes through the onnx-typed API since ORT's plain
    numpy path rejects it."""
    onnx_type = _bf16_onnx_type(arr.dtype)
    if onnx_type is None:
        return ort.OrtValue.ortvalue_from_numpy(arr)
    return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(arr, onnx_type)


def to_tt(arr: np.ndarray) -> ort.OrtValue:
    """Copy a host numpy array (any EP dtype, incl. ml_dtypes.bfloat16) into a
    device-resident OrtValue on the TT device, and return it — e.g. to bind as an
    IOBinding input the timed loop reuses without re-uploading."""
    onnx_type = _bf16_onnx_type(arr.dtype)
    dev = ort.OrtValue.ortvalue_from_shape_and_type(
        list(arr.shape),
        arr.dtype if onnx_type is None else onnx_type,
        memory_info=tt_memory_info(),
    )
    ort.copy_tensors([host_ortvalue(arr)], [dev])
    return dev


def tt_empty(shape, dtype) -> ort.OrtValue:
    """An uninitialized device-resident OrtValue on the TT device — e.g. an
    IOBinding output the run writes in place. `dtype` is a numpy dtype (incl.
    ml_dtypes.bfloat16) or an int ONNX element type (onnx.TensorProto.*)."""
    elem = dtype if isinstance(dtype, int) else (_bf16_onnx_type(dtype) or dtype)
    return ort.OrtValue.ortvalue_from_shape_and_type(
        list(shape), elem, memory_info=tt_memory_info()
    )


def from_tt(ov: ort.OrtValue) -> np.ndarray:
    """Drain a device OrtValue to host and return a numpy array. bf16 comes back
    as an ml_dtypes.bfloat16 array (cast with `.astype(np.float32)` to compare) —
    ORT can't hand bf16 straight to numpy, so we drain into an aliased array."""
    out = np.zeros(list(ov.shape()), dtype=_numpy_dtype(ov))
    ort.copy_tensors([ov], [host_ortvalue(out)])  # host_ortvalue aliases `out`
    return out
