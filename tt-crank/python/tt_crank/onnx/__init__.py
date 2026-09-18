# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime front door for the tt-crank execution provider.

``register()`` points ONNX Runtime at the plugin library and, like
``tt_crank.torch``, unregisters it again at interpreter exit so the device mesh
closes before C-runtime teardown. ``session()`` builds a session on the EP;
``to_device`` / ``empty`` / ``to_host`` / ``bind`` / ``run`` move tensors to and
from the device. Model builders, random data and CPU goldens round it out for
tests and quick experiments.
"""

import atexit
import os
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper, shape_inference

from tt_crank._runtime_env import setup_tt_metal_home

EP_NAME = "TTCrankExecutionProvider"
_REGISTRATION_NAME = "tt_crank"
_registered = False
EP_OPSET_VERSION = 22


class DType:
    """ONNX element types (onnx.TensorProto.*)."""

    F32 = onnx.TensorProto.FLOAT
    F64 = onnx.TensorProto.DOUBLE
    BF16 = onnx.TensorProto.BFLOAT16
    I32 = onnx.TensorProto.INT32
    I64 = onnx.TensorProto.INT64
    BOOL = onnx.TensorProto.BOOL


def library_path() -> Path:
    """The plugin library: wheel payload if installed, else the tt-mlir build tree
    (BUILD_DIR overrides <tt-mlir>/build)."""
    wheel_lib = Path(__file__).resolve().parents[1] / "lib" / "libtt_crank_ort.so"
    if wheel_lib.exists():
        return wheel_lib

    tt_mlir_root = Path(__file__).resolve().parents[4]
    build_dir = Path(os.environ.get("BUILD_DIR", tt_mlir_root / "build"))
    return build_dir / "tt-crank" / "src" / "onnx" / "libtt_crank_ort.so"


def register() -> None:
    """Register the EP library with ONNX Runtime (idempotent)."""
    global _registered
    if _registered:
        return
    path = library_path()
    if not path.exists():
        raise FileNotFoundError(f"tt-crank ONNX EP library not found: {path}")

    setup_tt_metal_home()
    onnxruntime.register_execution_provider_library(_REGISTRATION_NAME, str(path))
    atexit.register(
        onnxruntime.unregister_execution_provider_library, _REGISTRATION_NAME
    )
    _registered = True


def device() -> onnxruntime.OrtEpDevice:
    """The TT EP device (requires register())."""
    return next(d for d in onnxruntime.get_ep_devices() if d.ep_name == EP_NAME)


def session(
    model: str | Path | bytes,
    compile_options: dict[str, str] | None = None,
    allow_cpu_fallback: bool = False,
    free_dims: dict[str, int] | None = None,
    ep_context: bool | str | Path = False,
    options: onnxruntime.SessionOptions | None = None,
) -> onnxruntime.InferenceSession:
    """An InferenceSession on the TT EP.

    `compile_options` keys are CompileOptions field names. `free_dims` pins symbolic
    dims by name. `ep_context` enables EPContext export: True writes <model>_ctx.onnx
    next to a file model, a path writes there. `options` seeds the SessionOptions.
    """
    options = options or onnxruntime.SessionOptions()
    options.add_provider_for_devices([device()], compile_options or {})

    if not allow_cpu_fallback:
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    if ep_context:
        options.add_session_config_entry("ep.context_enable", "1")
        if ep_context is not True:
            options.add_session_config_entry("ep.context_file_path", str(ep_context))

    for name, value in (free_dims or {}).items():
        options.add_free_dimension_override_by_name(name, value)

    return onnxruntime.InferenceSession(_as_model(model), sess_options=options)


def _as_model(model: str | Path | bytes) -> str | bytes:
    return str(model) if isinstance(model, Path) else model


# ---- device tensors -----------------------------------------------------------


def memory_info() -> onnxruntime.OrtMemoryInfo:
    """DEFAULT memory info of the TT EP device, for allocating device OrtValues."""
    return device().memory_info(onnxruntime.OrtDeviceMemoryType.DEFAULT)


def _is_bf16(dtype) -> bool:
    return np.dtype(dtype) == np.dtype(ml_dtypes.bfloat16)


def _numpy_dtype(ort_type: str) -> type:
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


def _host_value(arr: np.ndarray) -> onnxruntime.OrtValue:
    """A host OrtValue aliasing `arr`."""
    if _is_bf16(arr.dtype):
        return onnxruntime.OrtValue.ortvalue_from_numpy_with_onnx_type(arr, DType.BF16)
    return onnxruntime.OrtValue.ortvalue_from_numpy(arr)


def empty(shape, dtype) -> onnxruntime.OrtValue:
    """An uninitialized OrtValue on the TT device."""
    elem = DType.BF16 if _is_bf16(dtype) else np.dtype(dtype)
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        list(shape), elem, memory_info=memory_info()
    )


def to_device(arr: np.ndarray) -> onnxruntime.OrtValue:
    """Copy `arr` into a new OrtValue on the TT device."""
    dev = empty(arr.shape, arr.dtype)
    onnxruntime.copy_tensors([_host_value(arr)], [dev])
    return dev


def to_host(value: onnxruntime.OrtValue) -> np.ndarray:
    """Copy a device OrtValue back to a numpy array (bf16 as ml_dtypes.bfloat16)."""
    out = np.zeros(list(value.shape()), dtype=_numpy_dtype(value.data_type()))
    onnxruntime.copy_tensors([value], [_host_value(out)])  # _host_value aliases `out`
    return out


def bind(
    session: onnxruntime.InferenceSession, inputs: dict[str, np.ndarray]
) -> tuple[onnxruntime.IOBinding, list[onnxruntime.OrtValue]]:
    """Upload `inputs` and allocate device outputs: (io_binding, device_outputs).
    Run with session.run_with_iobinding(io) as often as needed, drain with to_host."""
    io = session.io_binding()
    for name, arr in inputs.items():
        io.bind_ortvalue_input(name, to_device(arr))

    outputs = []
    for out in session.get_outputs():
        if not all(isinstance(d, int) for d in out.shape):
            raise ValueError(f"{out.name}: symbolic shape {out.shape}")
        value = empty(out.shape, _numpy_dtype(out.type))
        io.bind_ortvalue_output(out.name, value)
        outputs.append(value)

    return io, outputs


def run(
    session: onnxruntime.InferenceSession, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """One run with device-resident inputs and outputs."""
    io, outputs = bind(session, inputs)
    session.run_with_iobinding(io)
    return [to_host(value) for value in outputs]


# ---- model building ------------------------------------------------------------


def numpy_dtype(dtype: int) -> np.dtype:
    """numpy dtype for an ONNX element type (DType.*)."""
    return helper.tensor_dtype_to_np_dtype(dtype)


def make_node(
    op: str, inputs: list[str], outputs: list[str], **attrs
) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def vi(name: str, shape: list | None, dtype: int = DType.F32) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def make_tensor(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name)


def make_model(
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto] = (),
    opset: int = EP_OPSET_VERSION,
) -> bytes:
    graph = helper.make_graph(
        list(nodes),
        "ep_test",
        list(inputs),
        list(outputs),
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    return shape_inference.infer_shapes(model).SerializeToString()


# ---- random data --------------------------------------------------------------

_RNG = np.random.default_rng(0)


def reseed(seed: int = 0) -> None:
    global _RNG
    _RNG = np.random.default_rng(seed)


def randn(*shape: int, dtype=np.float32) -> np.ndarray:
    return _RNG.standard_normal(shape).astype(dtype)


def randint(*shape: int, low: int = 1, high: int = 9, dtype=np.int32) -> np.ndarray:
    return _RNG.integers(low, high, size=shape).astype(dtype)


# ---- goldens ------------------------------------------------------------------


def session_opts() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def cpu_golden(
    model: str | Path | bytes, inputs: dict[str, np.ndarray]
) -> list[np.ndarray]:
    model = str(model) if isinstance(model, Path) else model
    providers = ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(model, session_opts(), providers=providers).run(
        None, inputs
    )


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two arrays, flattened."""
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def assert_tt_matches_cpu(
    model: str | Path | bytes,
    inputs: dict[str, np.ndarray],
    atol: float = 2e-2,
    rtol: float = 2e-2,
    **session_kwargs,
) -> None:
    """Run `model` on the TT EP (device-resident I/O) and compare with the CPU EP."""
    got = run(session(model, options=session_opts(), **session_kwargs), inputs)
    want = cpu_golden(model, inputs)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=atol, rtol=rtol)
