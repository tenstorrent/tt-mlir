# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers for the ONNX Runtime plugin-EP tests."""

import os
import pathlib

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

TT_MLIR_ROOT = pathlib.Path(__file__).resolve().parents[4]
EP_NAME = "TTKurblaExecutionProvider"
REGISTRATION_NAME = "tt_kurbla"
EP_OPSET_VERSION = 22


class DType:
    """ONNX element types (onnx.TensorProto.*)."""

    F32 = onnx.TensorProto.FLOAT
    F64 = onnx.TensorProto.DOUBLE
    BF16 = onnx.TensorProto.BFLOAT16
    I32 = onnx.TensorProto.INT32
    I64 = onnx.TensorProto.INT64
    BOOL = onnx.TensorProto.BOOL


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
    ep_context: bool | str | pathlib.Path = False,
) -> ort.InferenceSession:
    """A session with the tt EP. `free_dims` pins symbolic dims by name. `ep_context` enables
    EPContext export: True writes <model>_ctx.onnx next to a file model, a path writes there."""
    options = session_opts()
    options.add_provider_for_devices([tt_device()], compile_options or {})

    if not allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    if ep_context:
        options.add_session_config_entry("ep.context_enable", "1")
        if ep_context is not True:
            options.add_session_config_entry("ep.context_file_path", str(ep_context))

    for name, value in (free_dims or {}).items():
        options.add_free_dimension_override_by_name(name, value)

    return ort.InferenceSession(_as_model(model), sess_options=options)


def cpu_golden(
    model: str | pathlib.Path | bytes, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    session = ort.InferenceSession(
        _as_model(model), session_opts(), providers=["CPUExecutionProvider"]
    )
    return session.run(None, inputs)


# ---- model-building helpers --------------------------------------------------

_RNG = np.random.default_rng(0)


def reseed(seed: int = 0) -> None:
    global _RNG
    _RNG = np.random.default_rng(seed)


def numpy_dtype(dtype: int) -> np.dtype:
    """numpy dtype for an ONNX element type (DType.*)."""
    return helper.tensor_dtype_to_np_dtype(dtype)


def make_node(
    op: str, inputs: list[str], outputs: list[str], **attrs
) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def vi(name, shape, dtype=DType.F32):
    return helper.make_tensor_value_info(name, dtype, shape)


def make_model(
    nodes, inputs, outputs, initializers=(), opset=EP_OPSET_VERSION
) -> bytes:
    graph = helper.make_graph(
        list(nodes),
        "ep_op_test",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return shape_inference.infer_shapes(model).SerializeToString()


def randn(*shape: int, dtype=np.float32) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(dtype)


def randint(*shape: int, low: int = 1, high: int = 9, dtype=np.int32) -> np.ndarray:
    return _RNG.integers(low, high, size=shape).astype(dtype)


def make_tensor(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name)


def assert_tt_matches_cpu(
    model: bytes, inputs: dict, atol: float = 2e-2, rtol: float = 2e-2, **session_kwargs
) -> None:
    got = run_on_tt(session_on_tt(model, **session_kwargs), inputs)
    want = cpu_golden(model, inputs)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=atol, rtol=rtol)


# ---- device tensor helpers ---------------------------------------------------
# numpy-backed OrtValues moved to/from the TT device; bf16 uses ml_dtypes.bfloat16.


def tt_memory_info():
    """DEFAULT memory info of the TT EP device, for allocating device OrtValues."""
    return tt_device().memory_info(ort.OrtDeviceMemoryType.DEFAULT)


def _is_bf16(dtype) -> bool:
    return np.dtype(dtype) == np.dtype(ml_dtypes.bfloat16)


def _numpy_dtype(ort_type: str):
    """numpy dtype for an ORT type string such as "tensor(bfloat16)"."""
    name = ort_type[len("tensor(") : -1]
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
    """A host OrtValue aliasing `arr`. numpy has no bf16, so bf16 goes by ONNX element type."""
    if _is_bf16(arr.dtype):
        return ort.OrtValue.ortvalue_from_numpy_with_onnx_type(arr, DType.BF16)
    return ort.OrtValue.ortvalue_from_numpy(arr)


def tt_empty(shape, dtype) -> ort.OrtValue:
    """An uninitialized OrtValue on the TT device."""
    elem = DType.BF16 if _is_bf16(dtype) else np.dtype(dtype)
    return ort.OrtValue.ortvalue_from_shape_and_type(
        list(shape), elem, memory_info=tt_memory_info()
    )


def to_tt(arr: np.ndarray) -> ort.OrtValue:
    """Copy `arr` into a new OrtValue on the TT device."""
    dev = tt_empty(arr.shape, arr.dtype)
    ort.copy_tensors([host_ortvalue(arr)], [dev])
    return dev


def from_tt(ov: ort.OrtValue) -> np.ndarray:
    """Copy a device OrtValue back to a numpy array (bf16 as ml_dtypes.bfloat16)."""
    out = np.zeros(list(ov.shape()), dtype=_numpy_dtype(ov.data_type()))
    ort.copy_tensors([ov], [host_ortvalue(out)])  # host_ortvalue aliases `out`
    return out


def bind_on_tt(session: ort.InferenceSession, inputs: dict[str, np.ndarray]):
    """Upload `inputs` and allocate device outputs; returns (io_binding, device_outputs).
    Run with session.run_with_iobinding(io) as often as needed, drain with from_tt."""
    io = session.io_binding()
    for name, arr in inputs.items():
        io.bind_ortvalue_input(name, to_tt(arr))
    outputs = []
    for out in session.get_outputs():
        assert all(
            isinstance(d, int) for d in out.shape
        ), f"{out.name}: symbolic shape {out.shape}"
        ov = tt_empty(out.shape, _numpy_dtype(out.type))
        io.bind_ortvalue_output(out.name, ov)
        outputs.append(ov)
    return io, outputs


def run_on_tt(
    session: ort.InferenceSession, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """One run with device-resident inputs and outputs."""
    io, outputs = bind_on_tt(session, inputs)
    session.run_with_iobinding(io)
    return [from_tt(ov) for ov in outputs]
